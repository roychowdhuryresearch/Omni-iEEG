from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
class EventInferenceDataset(Dataset):
    
    def __init__(self, npz_path, flip=False):
        self.npz_path = npz_path
        loaded = np.load(npz_path, allow_pickle=True)
        self.start = loaded["start"]
        self.end = loaded["end"]
        self.real_start = loaded["real_start"]
        self.real_end = loaded["real_end"]
        self.waveforms = loaded["hfo_waveforms"]
        self.name = loaded["name"]
        self.is_boundary = loaded["is_boundary"]
        self.file_name = loaded["file_name"]
        self.detector = loaded["detector"]
        self.participant = loaded["participant"]
        self.session = loaded["session"]
        self.file_name = loaded["file_name"]
        
        
        self.length = len(self.waveforms)
        self.flip = flip
       
    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """
        start = self.start[ind]
        end = self.end[ind]
        name = self.name[ind]
        waveform = self.waveforms[ind]
        real_start = self.real_start[ind]
        real_end = self.real_end[ind]
        is_boundary = self.is_boundary[ind]
        participant = self.participant[ind]
        session = self.session[ind]
        file_name = self.file_name[ind]
        
        waveform = torch.from_numpy(waveform)
        if self.flip:
            chance = random.random()
            if chance < 0.5:
                waveform = torch.flip(waveform, [0])
        
        # Package all data into a dictionary
        sample = {
            'start': start,
            'end': end,
            'name': name,
            'waveform': waveform,
            'real_start': real_start,
            'real_end': real_end,
            'is_boundary': is_boundary,
            'participant': participant,
            'session': session,
            'file_name': file_name
        }
        
        return sample
        

        # feature = torch.from_numpy(self.feature[ind])#[:, index+40:index+184]
        # label = self.label[ind]
        # channel_names = self.channel_names[ind]
        # start_end = self.start_end[ind]
        
        # chance = random.random()
        # if self.flip and chance < 0.5:
        #     feature = torch.flip(feature, [2])
        # return self.patient_name,feature, label, channel_names, start_end
