import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler, random_split

from omni_ieeg.event_model.train.dataloader_parquet import EventDataset_3label_parquet
from omni_ieeg.event_model.train.model_3label.configs import model_preprocessing_configs
from omni_ieeg.event_model.train.model_3label.comapre_parameters import count_parameters
from torch.utils.data import  DataLoader, ConcatDataset
import copy
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import glob
import re
import argparse
import argparse

def collate_events(batch):
    """Collate function to handle dictionary outputs from EventDataset."""
    # Separate waveforms and other data
    waveforms = torch.stack([item['waveform'] for item in batch]).float()  # Convert to float32
    
    # Changed: Convert labels to long tensor for multi-class classification
    # For multi-class classification, we need integer labels, not float
    # CrossEntropyLoss expects class indices as integers, not one-hot encoded vectors
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # Collect metadata into lists
    metadata = {key: [item[key] for item in batch] 
                for key in batch[0].keys() if key not in ['waveform', 'label']}
    
    return {
        'waveform': waveforms,
        'label': labels,
        'metadata': metadata
    }

# Helper function for accuracy
def calculate_accuracy(outputs, labels):
    """
    Calculates multi-class classification accuracy from logits.
    
    Changed: We now use argmax to find the predicted class with the highest probability
    instead of thresholding which only works for binary classification.
    """
    # Get the predicted class with highest score
    _, predicted = torch.max(outputs, 1)
    # Count correct predictions
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

class Trainer():
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
            
        self._init_data()
        self._init_model()
        self._initialize_optimizer()
        self._setup_dataloaders()

        # Calculate class weights based on class distribution if specified
        if self.training_config.get("use_class_weights", True):
            self.class_weights = self._compute_class_weights()
            if self.verbose:
                print(f"Using class weights: {self.class_weights}")
        else:
            self.class_weights = None
            
        # Changed: Define CrossEntropyLoss with class weights to handle imbalanced data
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction="none"
        ).to(self.device)

    def _init_data(self):
        training_files = glob.glob(os.path.join(self.data_config["feature_path"], "*.parquet"))
        testing_files = glob.glob(os.path.join(self.data_config["test_feature_path"], "*.parquet"))
        self.training_dataset = []
        self.testing_dataset = []
        for train_file in training_files:
            new_dataset = EventDataset_3label_parquet(train_file, flip=self.training_config["flip"])
            self.training_dataset.append(new_dataset)
        for test_file in testing_files:
            new_dataset = EventDataset_3label_parquet(test_file, flip=False)
            self.testing_dataset.append(new_dataset)
        self.training_dataset = ConcatDataset(self.training_dataset)
        self.testing_dataset = ConcatDataset(self.testing_dataset)
        if self.verbose: print(f"Total training samples: {len(self.training_dataset)}, Total testing samples: {len(self.testing_dataset)}")

    def _init_model(self):
        self.model = self.model_config["model_class"](**self.model_config["model_params"])
        self.model.to(self.training_config["device"])
        self.pre_processing = self.pre_processing_config["preprocessing_class"](**self.pre_processing_config["preprocessing_params"])
    def _initialize_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.training_config["learning_rate"])

    def _setup_dataloaders(self):
        """Constructs training, validation, and testing dataloaders."""
        batch_size = self.training_config["batch_size"]
        num_workers = self.training_config.get("num_workers", 0)
        val_split_ratio = self.training_config.get("validation_split_ratio", 0.2)
        use_weighted_sampling = self.training_config.get("use_weighted_sampling", False)
        
        total_train_size = len(self.training_dataset)
        if total_train_size == 0:
            raise ValueError("Training dataset is empty. Check _init_data.")
            
        val_size = int(total_train_size * val_split_ratio)
        train_size = total_train_size - val_size
        
        if train_size <= 0 or val_size < 0:
             raise ValueError(f"Invalid train/validation split sizes: train={train_size}, val={val_size}. Check dataset size and split ratio.")

        if self.verbose: print(f"Splitting training data: {train_size} train, {val_size} validation")

        generator = torch.Generator().manual_seed(self.training_config.get('seed', 42))
        train_subset, val_subset = random_split(self.training_dataset, [train_size, val_size], generator=generator)
        
        # Option to use weighted sampling for training to handle class imbalance
        sampler = None
        if use_weighted_sampling:
            # Calculate class weights for sampling
            sample_weights = self._compute_sample_weights(train_subset)
            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_subset),
                replacement=True
            )
            if self.verbose:
                print("Using weighted sampling for training data")
                
        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=(sampler is None),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_events,
            pin_memory=True if self.device == 'cuda' else False,
        )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_events,
            pin_memory=True if self.device == 'cuda' else False
        )

        
        self.test_loader = DataLoader(
            self.testing_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_events,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        if self.verbose: print("DataLoaders created.")
        
    def _compute_sample_weights(self, subset):
        """
        Compute sampling weights for each sample to balance class distribution.
        More weight given to samples from minority classes.
        """
        # Count classes
        label_counts = {0: 0, 1: 0, 2: 0}
        all_labels = []
        
        # First count total class distribution in the subset
        for idx in subset.indices:
            # ConcatDataset doesn't have _get_indices, we need to find which dataset this index belongs to
            dataset_idx = 0
            sample_idx = idx
            
            # Find which dataset the index belongs to
            for i, dataset in enumerate(self.training_dataset.datasets):
                dataset_len = len(dataset)
                if sample_idx < dataset_len:
                    dataset_idx = i
                    break
                sample_idx -= dataset_len
            
            # Now get the label from the correct dataset and sample
            dataset = self.training_dataset.datasets[dataset_idx]
            label = dataset.label[sample_idx]
            
            label_counts[label] = label_counts.get(label, 0) + 1
            all_labels.append(label)
            
        # Calculate class weights (inverse of frequency)
        class_weights = {}
        n_samples = len(subset)
        n_classes = len(label_counts)
        
        for label, count in label_counts.items():
            if count > 0:
                class_weights[label] = n_samples / (n_classes * count)
            else:
                class_weights[label] = 0
                
        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for label in all_labels]
        
        if self.verbose:
            print(f"Class distribution in training subset: {label_counts}")
            print(f"Class weights for sampling: {class_weights}")
            
        return torch.tensor(sample_weights, dtype=torch.float)

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        # Enable training-specific preprocessing if applicable (e.g., random shifts)
        if hasattr(self.pre_processing, 'train'):
            self.pre_processing.train() # Set preprocessing to train mode if it has one
        if hasattr(self.pre_processing, 'enable_random_shift'):
            self.pre_processing.enable_random_shift()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_loader)

        # Use tqdm for progress bar if not verbose (or always if preferred)
        train_iterator = tqdm(self.train_loader, desc="Training Epoch", leave=False, disable=not self.verbose)
        all_outputs = []
        all_labels = []
        for batch in train_iterator:
            waveforms = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device) # Labels should be [batch_size]

            # Apply preprocessing
            processed_waveforms = self.pre_processing(waveforms)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Changed: For CrossEntropyLoss, model outputs should be [batch_size, num_classes]
            # without any activation function like sigmoid
            outputs = self.model(processed_waveforms) # Expected shape [batch_size, 3]

            # Changed: CrossEntropyLoss expects raw logits, not squeezed outputs
            # Labels should be class indices
            loss = self.criterion(outputs, labels)

            batch_loss = loss.mean().item()  # Get mean loss for reporting
            batch_accuracy = calculate_accuracy(outputs, labels)

            # Reduce the loss before backprop
            loss_reduced = loss.mean()  # Manually reduce to a scalar
            loss_reduced.backward()  # Now backward() works on a scalar
            
            self.optimizer.step()

            # Calculate metrics
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

            # Update tqdm description
            train_iterator.set_postfix(loss=batch_loss, acc=batch_accuracy)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # Calculate accuracy
        avg_accuracy = calculate_accuracy(all_outputs, all_labels)
        return avg_loss, avg_accuracy, all_outputs, all_labels

    def _evaluate(self, dataloader):
        """Runs evaluation on a given dataloader (validation or test)."""
        self.model.eval()
        # Disable training-specific preprocessing if applicable
        if hasattr(self.pre_processing, 'eval'):
             self.pre_processing.eval() # Set preprocessing to eval mode if it has one
        if hasattr(self.pre_processing, 'disable_random_shift'):
            self.pre_processing.disable_random_shift()

        total_loss = 0.0
        total_accuracy = 0.0
        all_outputs = []
        all_labels = []
        all_metadata = [] # To store metadata if needed

        num_batches = len(dataloader)

        # Use tqdm for progress bar
        eval_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not self.verbose)

        with torch.no_grad():
            for batch in eval_iterator:
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label'].to(self.device)

                # Apply preprocessing
                processed_waveforms = self.pre_processing(waveforms)

                # Forward pass
                # Changed: For multi-class, output shape is [batch_size, num_classes]
                outputs = self.model(processed_waveforms)

                # Calculate loss
                # Changed: CrossEntropyLoss expects raw logits and class indices
                loss = self.criterion(outputs, labels)
                total_loss += loss.mean().item()

                # Calculate accuracy
                batch_accuracy = calculate_accuracy(outputs, labels)
                total_accuracy += batch_accuracy

                # Store outputs, labels, and metadata for potential later analysis
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                all_metadata.append(batch['metadata']) # Store metadata dict

                eval_iterator.set_postfix(loss=loss.mean().item(), acc=batch_accuracy)

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        # Concatenate results from all batches
        all_outputs = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty(0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0)
        
        # Combine metadata
        combined_metadata = {k: [item for sublist in [m[k] for m in all_metadata] for item in sublist]
                            for k in all_metadata[0].keys()} if all_metadata else {}
                            
        # Changed: Store both predicted class and probabilities for all classes
        # Get predicted class (0, 1, or 2)
        _, pred_classes = torch.max(all_outputs, 1)
        pred_classes = pred_classes.numpy()
        
        # Get probabilities via softmax
        probs = torch.nn.functional.softmax(all_outputs, dim=1).numpy()
        
        # Store in metadata
        true_values = all_labels.numpy()
        combined_metadata["predicted_class"] = pred_classes
        combined_metadata["true_class"] = true_values
        
        # Store probability for each class
        combined_metadata["prob_class0"] = probs[:, 0]  # Normal
        combined_metadata["prob_class1"] = probs[:, 1]  # Artifact only
        combined_metadata["prob_class2"] = probs[:, 2]  # Spike

        return avg_loss, avg_accuracy, all_outputs, all_labels, combined_metadata

    def train(self):
        """Main training loop."""
        since = time.time()
        best_loss = float('inf')
        best_model_path = None
        output_dir = self.data_config.get("output_dir", ".") # Use output_dir from config
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        num_epochs = self.training_config.get("num_epochs", 10) # Get num_epochs from config

        if self.verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Output directory: {output_dir}")

        for epoch in range(num_epochs):
            # Train one epoch
            train_loss, train_acc, _, _ = self._train_epoch()

            # Validate
            val_loss, val_acc, _, _, _ = self._evaluate(self.val_loader) # Ignore outputs/labels/metadata for now

            if self.verbose:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print("-" * 20)

            # Save the best model based on validation loss
            if self.val_loader and val_loss < best_loss:
                best_loss = val_loss
                best_model_path = os.path.join(output_dir, "best_model.pth")
                # Save model state_dict and potentially preprocessing state if needed
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                    # Include preprocessing state if it's stateful and needs saving
                    # 'preprocessing_state_dict': self.pre_processing.state_dict() if isinstance(self.pre_processing, nn.Module) else None
                }
                torch.save(checkpoint, best_model_path)
                if self.verbose:
                    print(f"New best model saved to {best_model_path} (Val Loss: {best_loss:.4f})")

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best Validation Loss: {best_loss:.4f}")
        if best_model_path:
             print(f"Best model saved at: {best_model_path}")
        else:
             print("No best model saved (validation loader might be missing or no improvement).")

        return best_model_path

    def test(self, model_path=None):
        """Evaluate the model on the test set."""
        if model_path is None:
            # Default to loading the best model from the training output directory
             model_path = os.path.join(self.data_config.get("output_dir", "."), "best_model.pth")

        if not os.path.exists(model_path):
            print(f"Error: Model checkpoint not found at {model_path}")
            return None, None, None, None, None # Return Nones matching the evaluate return signature

        if self.verbose: print(f"Loading model for testing from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Optionally load optimizer and preprocessing state if needed for resuming, but not typically for testing

        if self.verbose: print("Starting evaluation on the test set...")
        test_loss, test_acc, test_outputs, test_labels, test_metadata = self._evaluate(self.test_loader)

        if self.verbose:
            print(f"Test Results | Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        # Ensure all arrays in the metadata are 1D before creating DataFrame
        for key, values in test_metadata.items():
            if isinstance(values, np.ndarray) and values.ndim > 1:
                test_metadata[key] = values.flatten()

        # save test_metadata to a csv file
        test_metadata_path = os.path.join(self.data_config.get("output_dir", "."), "test_metadata.csv")
        pd.DataFrame(test_metadata).to_csv(test_metadata_path, index=False)
        
        return test_loss, test_acc, test_outputs, test_labels, test_metadata

    def _compute_class_weights(self):
        """Compute class weights based on class frequencies in training data"""
        # Calculate class distribution from training dataset
        label_counts = {0: 0, 1: 0, 2: 0}  # Initialize counts for each class
        
        # Count labels directly from each dataset in the ConcatDataset
        for dataset in self.training_dataset.datasets:
            # Each dataset is an EventDataset_3label with a label array
            unique_labels, counts = np.unique(dataset.label, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_counts[label] = label_counts.get(label, 0) + count
            
        total_samples = sum(label_counts.values())
        
        # Compute inverse frequency weighting
        # More weight to less frequent classes
        weights = torch.zeros(len(label_counts))
        for label, count in label_counts.items():
            if count > 0:
                weights[label] = total_samples / (len(label_counts) * count)
        
        if self.verbose:
            print(f"Class distribution: {label_counts}")
            print(f"Computed weights: {weights}")
            
        return weights.to(self.device)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["cnn", "vit", "lstm", "transformer", "timesnet"], help="Model name (cnn, vit, lstm, transformer, timesnet)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--training_feature_path", type=str, required=True)
    parser.add_argument("--testing_feature_path", type=str, required=True)
    parser.add_argument("--time_window_ms", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split_ratio", type=float, default=0.2)
    parser.add_argument("--use_class_weights", type=bool, default=False)
    parser.add_argument("--use_weighted_sampling", type=bool, default=True)
    parser.add_argument("--flip", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    
    
    
    args = parser.parse_args()
    
    model_name = args.model_name
    device = args.device
    output_dir = args.output_dir
    # Create date-based subfolder
    base_output_dir = f"{output_dir}/{model_name}_3label/"
    today_date = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_output_dir, today_date)
    
    # Check existing run folders for today
    if os.path.exists(date_dir):
        existing_runs = glob.glob(os.path.join(date_dir, "run*"))
        run_numbers = [int(re.search(r'run(\d+)', run).group(1)) for run in existing_runs if re.search(r'run(\d+)', run)]
        next_run = max(run_numbers, default=0) + 1
    else:
        next_run = 1
    
    # Create final output directory
    output_dir = os.path.join(date_dir, f"run{next_run}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model outputs to: {output_dir}")
    
    data_config = {
        "feature_path": args.training_feature_path,
        "test_feature_path": args.testing_feature_path,
        "output_dir": output_dir,
        "time_window_ms": args.time_window_ms
    }
    training_config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'device': device,
        'verbose': True,
        "flip": args.flip,
        "num_workers": args.num_workers,
        "validation_split_ratio": args.validation_split_ratio,
        "use_class_weights": args.use_class_weights,  # Enable class weighting to handle imbalanced classes
        "use_weighted_sampling": args.use_weighted_sampling  # Enable weighted sampling for training
    }
    model_config = model_preprocessing_configs[model_name]["model_config"]
    pre_processing_config = model_preprocessing_configs[model_name]["preprocessing_config"]
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
    
    # Display selected model information
    print(f"\n=== Selected Model: {model_config['model_class'].__name__} ===")
    print(f"Model Parameters: {model_config['model_params']}")
    print(f"Preprocessing: {pre_processing_config['preprocessing_class'].__name__}")
    print(f"Class Weighting: {'Enabled' if training_config['use_class_weights'] else 'Disabled'}")
    print(f"Weighted Sampling: {'Enabled' if training_config['use_weighted_sampling'] else 'Disabled'}")
    
    # Initialize model to count parameters
    temp_model = model_config["model_class"](**model_config["model_params"])
    param_count = count_parameters(temp_model)
    print(f"Trainable Parameters: {param_count:,}")
    
    trainer = Trainer(data_config, model_config, pre_processing_config, training_config)
    
    # --- Start Training ---
    best_model_file = trainer.train()

    # --- Run Testing ---
    if best_model_file:
         print("\n--- Testing Best Model ---")
         trainer.test(model_path=best_model_file)
    else:
         print("\nSkipping testing as no best model was saved.")


     