from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
from scipy.signal import resample_poly

class ChannelDataset(Dataset):
    
    def __init__(self, npz_path, label_name="artifact", flip=True, downsample=0):
        self.npz_path = npz_path
        loaded = np.load(npz_path, allow_pickle=True)
        self.data = loaded["data"]
        # downsample
        if downsample > 0:
            self.data = resample_poly(self.data, up=1, down=downsample, axis=-1)
        self.name = loaded["name"]
        self.labels = loaded["labels"]
        self.start_indices = loaded["start_indices"]
        self.end_indices = loaded["end_indices"]
        self.patient = loaded["patient"]
        self.edf_name = loaded["edf_name"]
        
        
        self.length = len(self.data)
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
        data = self.data[ind]
        name = self.name[ind]
        labels = self.labels[ind]
        start_indices = self.start_indices[ind]
        end_indices = self.end_indices[ind]
        patient = self.patient
        edf_name = self.edf_name
        
        data = torch.from_numpy(data)
        if self.flip:
            chance = random.random()
            if chance < 0.5:
                data = torch.flip(data, [0])
        
        # Package all data into a dictionary
        sample = {
            'data': data,
            'name': name,
            'labels': labels,
            'start_indices': start_indices,
            'end_indices': end_indices,
            'patient': patient,
            'edf_name': edf_name,
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
