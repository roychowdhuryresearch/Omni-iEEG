import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler, random_split

from omni_ieeg.channel_model.event_model_inference.dataloader import EventInferenceDataset
from omni_ieeg.event_model.train.model_3label.configs import model_preprocessing_configs
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
        feature_npzs = os.listdir(self.data_config["feature_path"])
        feature_npzs = [f for f in feature_npzs if f.endswith(".npz")]
        dfs = []
        for feature_npz in feature_npzs:
            feature_npz = os.path.join(self.data_config["feature_path"], feature_npz)
            assert os.path.exists(feature_npz), f"Feature npz {feature_npz} does not exist"
            print(f"Running inference on {feature_npz}")
            save_path = os.path.basename(feature_npz).replace(".npz", ".csv")
            save_path = os.path.join(self.data_config["output_dir"], save_path)
            df = self.run_one_inference(feature_npz, save_path)
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(os.path.join(self.data_config["output_dir"], "all_results.csv"), index=False)
        print(f"Saved all results to {os.path.join(self.data_config['output_dir'], 'all_results.csv')}")
                        
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
                _, predicted = torch.max(outputs, dim=1)
                

                # Store outputs, labels, and metadata for potential later analysis
                all_outputs.append(predicted.cpu())
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
        combined_metadata[f"3label_pred"] = pred_values
        df = pd.DataFrame(combined_metadata)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["cnn", "vit", "lstm", "transformer", "timesnet"], help="Model name (cnn, vit, lstm, transformer, timesnet)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to the feature npz files you get when running event_model/legacy_model_inference/hfo_features.py")
    parser.add_argument("--label_name", type=str, default="3label")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint you get when running event_model/train/train_3label.py")
    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    output_dir = args.output_dir
    feature_path = args.feature_path
    label_name = args.label_name
    model_path = args.model_path
    output_dir = os.path.join(output_dir, model_name, label_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model outputs to: {output_dir}")
    model_config = model_preprocessing_configs[model_name]["model_config"]
    model_config['model_path'] = model_path
    pre_processing_config = model_preprocessing_configs[model_name]["preprocessing_config"]
    data_config = {
        "feature_path": feature_path,
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