
import torch    
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from omni_ieeg.channel_model.channel_model_train.model.timesnet_layers.Embed import DataEmbedding
from omni_ieeg.channel_model.channel_model_train.model.timesnet_layers.Conv_Blocks import Inception_Block_V1  


def FFT_for_Period(x, k=2):
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N
    xf = torch.fft.rfft(x, dim=1) 

    # find period by amplitudes: here we assume that the periodic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, getting a list frequency_list with shape[T] 
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t)
    frequency_list = abs(xf).mean(0).mean(-1) 
    frequency_list[0] = 0

    #by torch.topk(),we can get the biggest k elements of frequency_list, and its positions(i.e. the k-main frequencies in top_list)
    _, top_list = torch.topk(frequency_list, k)

    #Returns a new Tensor 'top_list', detached from the current graph.
    #The result will never require gradient.Convert to a numpy instance
    top_list = top_list.detach().cpu().numpy()

    #period:a list of shape [top_k], recording the periods of mean frequencies respectively
    period = x.shape[1] // top_list

    #Here,the 2nd item returned has a shape of [B, top_k],representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list] 


class TimesBlock(nn.Module):
    def __init__(self, seq_len, k, d_model, d_ff, num_kernels):    ##configs is the configuration defined for TimesBlock
        super(TimesBlock, self).__init__() 
        self.seq_len = seq_len   ##sequence length 
        self.k = k    ##k denotes how many top frequencies are taken into consideration
        self.pred_len = 0  ## for classification
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                            num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                            num_kernels=num_kernels)
        )

    def forward(self, x):   
        B, T, N = x.size()
            #B: batch size  T: length of time series  N:number of features
        period_list, period_weight = FFT_for_Period(x, self.k)
            #FFT_for_Period() will be shown later. Here, period_list([top_k]) denotes 
            #the top_k-significant period and period_weight([B, top_k]) denotes its weight(amplitude)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding : to form a 2D map, we need total length of the sequence, plus the part 
            # to be predicted, to be divisible by the period, so padding is needed
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            
            #2D convolution to grap the intra- and inter- period information
            out = self.conv(out)

            # reshape back, similar to reshape
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            #truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]

        # adaptive aggregation
        #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1) 

        #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        
        #add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res
    

class TimesNet(nn.Module):
    def __init__(self, seq_len, d_model, top_k ,d_ff, e_layers, embed, freq, dropout, enc_in, num_class, num_kernels):
        super(TimesNet, self).__init__()
        #params init
        self.seq_len = seq_len
        self.task_name = "classification"

        #stack TimesBlock for e_layers times to form the main part of TimesNet, named model
        self.model = nn.ModuleList([TimesBlock(seq_len, top_k, d_model, d_ff, num_kernels)
                                    for _ in range(e_layers)])
        
        #embedding & normalization
        # enc_in is the encoder input size, the number of features for a piece of data
        # d_model is the dimension of embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers # num of encoder layers
        self.layer_norm = nn.LayerNorm(d_model)

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(
                d_model * seq_len, num_class)
            self.final_ac = nn.Sigmoid()

    def forward(self, x_enc):
        x_enc = x_enc.unsqueeze(-1)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)

        # zero-out padding embeddings:The primary role of x_mark_enc in the code is to 
        # zero out the embeddings for padding positions in the output tensor through 
        # element-wise multiplication, helping the model to focus on meaningful data 
        # while disregarding padding.
        # output = output * x_mark_enc.unsqueeze(-1)
        
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        logits = output # self.final_ac(output)
        return logits
    


class NeuralTimesNetPreProcessing():
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data