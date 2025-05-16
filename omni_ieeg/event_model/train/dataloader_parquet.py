from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import random
import pandas as pd
class EventDataset_3label_parquet(Dataset):
    
    def __init__(self, parquet_path, flip=True):
        self.parquet_path = parquet_path
        loaded = pd.read_parquet(parquet_path)
        self.waveforms = loaded["hfo_waveforms"]
        self.edf_file = loaded["edf_file"]
        self.detector = loaded["detector"]
        self.method = loaded["method"]
        
        # Check that both artifact and spike labels exist in the file
        if "artifact" not in loaded or "spike" not in loaded:
            raise ValueError(f"Both 'artifact' and 'spike' labels must exist in {parquet_path}")
        
        artifact_labels = loaded["artifact"]
        spike_labels = loaded["spike"]
        
        # Create 3-class labels:
        # 0: artifact=0, spike=0 - Normal
        # 1: artifact=1, spike=0 - Artifact
        # 2: artifact=1, spike=1 - Spike
        self.label = np.zeros_like(artifact_labels, dtype=np.int64)
        
        # Set label=1 where artifact=1 and spike=0
        mask_artifact_only = (artifact_labels == 1) & (spike_labels == 0)
        self.label[mask_artifact_only] = 1
        
        # Set label=2 where artifact=1 and spike=1
        mask_spike = (artifact_labels == 1) & (spike_labels == 1)
        self.label[mask_spike] = 2
        
        # Check for invalid combinations (artifact=0, spike=1)
        invalid_mask = (artifact_labels == 0) & (spike_labels == 1)
        if np.any(invalid_mask):
            print(
                f"Invalid combination found in {parquet_path}: artifact=0, spike=1 in "
                f"{np.sum(invalid_mask)} samples, setting to class 0"
            )
        self.label[invalid_mask] = 0
            
        self.length = len(self.waveforms)
        self.flip = flip
        
        print(f"Dataset {parquet_path} loaded with {self.length} samples:")
        print(f"  - Class 0 (Normal): {np.sum(self.label == 0)}")
        print(f"  - Class 1 (Artifact): {np.sum(self.label == 1)}")
        print(f"  - Class 2 (Spike): {np.sum(self.label == 2)}")
    
    def __len__(self):
        # Return the total number of data samples
        return self.length

    def __getitem__(self, ind):
        """Returns the sample and its 3-class label at the index 'ind'
        
        Params:
        -------
        - ind: (int) The index of the sample to get

        Returns:
        --------
        - A dictionary with sample data including the 3-class label
        """
        waveform = self.waveforms[ind]
        label = self.label[ind]
        edf_file = self.edf_file[ind]
        detector = self.detector[ind]
        method = self.method[ind]
        
        waveform = torch.from_numpy(waveform.copy())  
        if self.flip:
            chance = random.random()
            if chance < 0.5:
                waveform = torch.flip(waveform, [0])
        
        # Package all data into a dictionary
        sample = {
            'waveform': waveform,
            'label': label,
            'edf_file': edf_file,
            'detector': detector,
            'method': method
        }
        
        return sample
