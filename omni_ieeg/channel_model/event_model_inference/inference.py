import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler, random_split

from omni_ieeg.channel_model.event_model_inference.dataloader import EventInferenceDataset
from omni_ieeg.event_model.train.model.configs import model_preprocessing_configs
from torch.utils.data import  DataLoader, ConcatDataset
import copy
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import glob
import re

def collate_events(batch):
    """Collate function to handle dictionary outputs from EventDataset."""
    # Separate waveforms and other data
    waveforms = torch.stack([item['waveform'] for item in batch]).float()  # Convert to float32
    
    # Collect metadata into lists
    metadata = {key: [item[key] for item in batch] 
                for key in batch[0].keys() if key not in ['waveform']}
    
    return {
        'waveform': waveforms,
        'metadata': metadata
    }


class Inferencer():
    def __init__(self, data_config, model_config, pre_processing_config, training_config):
        self.data_config = data_config
        self.model_config = model_config
        self.pre_processing_config = pre_processing_config
        self.training_config = training_config
        self.device = self.training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = training_config.get("verbose", True)
        
        seed = self.training_config.get('seed', 0)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        self._init_model(model_path = self.model_config["model_path"])
        self._initialize_optimizer()

        # Define loss function
        self.run_all_inference()
    def _init_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self.model_config["model_class"](**self.model_config["model_params"])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.training_config["device"])
        self.pre_processing = self.pre_processing_config["preprocessing_class"](**self.pre_processing_config["preprocessing_params"])
    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.training_config["learning_rate"])

    def run_all_inference(self):
        """Runs inference on a given dataloader (validation or test)."""
        self.model.eval()
        # Disable training-specific preprocessing if applicable
        if hasattr(self.pre_processing, 'eval'):
             self.pre_processing.eval() # Set preprocessing to eval mode if it has one
        if hasattr(self.pre_processing, 'disable_random_shift'):
            self.pre_processing.disable_random_shift()
        data_split = json.load(open(self.data_config["split_json"]))
        for dataset, dataset_dict in data_split.items():
            if self.verbose: print(f"========== Processing {dataset} ==========")
            for outcome, patient_list in dataset_dict.items():
                if self.verbose: print(f"========== Processing {outcome}")
                for patient in tqdm(patient_list, desc=f"Processing patient for {dataset} {outcome}"):
                    patient_dir = os.path.join(self.data_config["feature_path"], patient)
                    assert os.path.exists(patient_dir), f"Patient directory {patient_dir} does not exist"
                    feature_npzs = glob.glob(f"{patient_dir}/*/*/*.npz")
                    for feature_npz in feature_npzs:
                        assert os.path.exists(feature_npz), f"Feature npz {feature_npz} does not exist"
                        save_path = feature_npz.replace(".npz", ".csv").replace(self.data_config["feature_path"], self.data_config["output_dir"])
                        self.run_one_inference(feature_npz, save_path)
                        
    def run_one_inference(self, npz_path, save_path):            
        dataset = EventInferenceDataset(npz_path)
        dataloader = DataLoader(dataset, 
                                batch_size=self.training_config["batch_size"], 
                                shuffle=False, 
                                num_workers=self.training_config["num_workers"],
                                collate_fn=collate_events)
        num_batches = len(dataloader)

        all_outputs = []
        all_metadata = []
        # Use tqdm for progress bar
        eval_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not self.verbose)
        with torch.no_grad():
            for batch in eval_iterator:
                waveforms = batch['waveform'].to(self.device)

                # Apply preprocessing
                processed_waveforms = self.pre_processing(waveforms)

                # Forward pass
                outputs = self.model(processed_waveforms)
                
                # Ensure outputs are at least 1-dimensional for consistent concatenation
                outputs = outputs.squeeze()
                if outputs.dim() == 0:  # If scalar (0-dimensional), add a dimension
                    outputs = outputs.unsqueeze(0)

                # Store outputs, labels, and metadata for potential later analysis
                all_outputs.append(outputs.cpu())
                all_metadata.append(batch['metadata']) # Store metadata dict

        # Concatenate results from all batches
        all_outputs = torch.cat(all_outputs, dim=0) if all_outputs and len(all_outputs) > 0 else torch.empty(0)
        # Combine metadata (this part might need adjustment based on how you want to use it)
        combined_metadata = {k: [item for sublist in [m[k] for m in all_metadata] for item in sublist]
                             for k in all_metadata[0].keys()} if all_metadata else {}
                             
        # Ensure outputs and labels are 1-dimensional before adding to metadata
        pred_values = all_outputs.squeeze().numpy()
        
        # Check if values are still multi-dimensional and flatten if needed
        if pred_values.ndim > 1:
            pred_values = pred_values.flatten()
        combined_metadata[f"{self.training_config['label_name']}_pred"] = pred_values
        df = pd.DataFrame(combined_metadata)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    model_name = "cnn" # cnn, vit, lstm, transformer, timesnet
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    label_name = "artifact" # artifact, spike
    model_path = "/mnt/SSD1/nipsdataset/event_training/model_output/cnn_artifact_cross_patient/2025-04-25/run1/best_model.pth"
    output_dir = f"/mnt/SSD1/nipsdataset/channel_training/new_event_model_preds/{model_name}_cross_patient/{label_name}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model outputs to: {output_dir}")
    model_config = model_preprocessing_configs[model_name]["model_config"]
    model_config['model_path'] = model_path
    pre_processing_config = model_preprocessing_configs[model_name]["preprocessing_config"]
    data_config = {
        "feature_path": "/mnt/SSD1/nipsdataset/training/hfo_feature",
        "split_json": "/mnt/SSD1/nipsdataset/omni-ieeg/omni_ieeg/channel_model/channel_data_split_UCLA.json",
        "output_dir": output_dir,
        "time_window_ms": 2000
    }
    training_config = {
        'num_epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.0003,
        'seed': 42,
        'device': device,
        'verbose': True,
        "label_name": label_name,
        "flip": False,
        "num_workers": 16,
        "validation_split_ratio": 0.2
    }
    
    # save configs to output_dir
    with open(os.path.join(output_dir, "data_config.json"), "w") as f:
        json.dump(data_config, f)
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        model_config_text = model_config.copy()
        model_config_text["model_class"] = model_config["model_class"].__name__
        json.dump(model_config_text, f)
    with open(os.path.join(output_dir, "pre_processing_config.json"), "w") as f:
        pre_processing_config_text = pre_processing_config.copy()
        pre_processing_config_text["preprocessing_class"] = pre_processing_config["preprocessing_class"].__name__
        json.dump(pre_processing_config_text, f)
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f)
    inferencer = Inferencer(data_config, model_config, pre_processing_config, training_config)