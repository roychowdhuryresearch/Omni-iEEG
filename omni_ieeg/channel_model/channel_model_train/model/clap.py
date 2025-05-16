from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor
import torch
import torch.nn as nn

class CLAPClassifier(nn.Module):
    def __init__(self, num_class=5):
        super().__init__()
        self.feature_extractor = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.feature_extractor.sampling_rate = 1000
        self.feature_extractor.chunk_length_s = 60
        self.feature_extractor.max_length_s = 60
        self.feature_extractor.nb_max_samples = 1 * 60 * 1000
        self.feature_extractor.n_mels = 64 
        self.feature_extractor.f_min = 0
        self.feature_extractor.f_max = 500
        self.feature_extractor.frequency_min = 0
        self.feature_extractor.frequency_max = 500
        self.feature_extractor.n_fft = 1024
        self.feature_extractor.fft_window_size = 1024
        self.feature_extractor.nb_frequency_bins = 513
        self.feature_extractor.hop_length = 100
        self.feature_extractor.nb_max_frames = 600
        self.feature_extractor.return_attention_mask = False
        # print(self.feature_extractor)

        self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")

        # Freeze all parameters except audio_model and projection (optional)
        for name, param in self.model.named_parameters():
            if not (name.startswith("audio_model") or name.startswith("audio_projection")):
                param.requires_grad = False

        # Get projection dim from model config
        proj_dim = self.model.config.projection_dim

        # Add simple binary classifier head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_class) 
        )

    def forward(self, inputs):
        inputs = inputs.detach().cpu().numpy()
        inputs = self.feature_extractor(inputs, sampling_rate=1000, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = self.classifier(outputs.audio_embeds)
        return logits

class NeuralClapPreProcessing():
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data