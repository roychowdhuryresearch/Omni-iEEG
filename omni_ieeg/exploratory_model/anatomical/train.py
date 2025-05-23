import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler, random_split

from omni_ieeg.channel_model.channel_model_train.dataloader import ChannelDataset
from omni_ieeg.exploratory_model.anatomical.configs import model_preprocessing_configs
from torch.utils.data import  DataLoader, ConcatDataset
import copy
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import glob
import re
from omni_ieeg.dataloader.datafilter import DataFilter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning)


def collate_events(batch):
    """Collate function to handle dictionary outputs from EventDataset."""
    # Separate waveforms and other data
    data = torch.stack([item['data'] for item in batch]).float()  # Convert to float32
    # Ensure label is long for CrossEntropyLoss
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long) 
    
    # Collect metadata into lists
    metadata = {key: [item[key] for item in batch] 
                for key in batch[0].keys() if key not in ['data', 'labels']}
    
    return {
        'data': data,
        'labels': labels,
        'metadata': metadata
    }

# Enhanced metrics calculation function for multi-class
def calculate_metrics(outputs, labels, return_all=False, num_classes=5):
    """
    Calculate multiple classification metrics for multi-class classification.
    
    Args:
        outputs: Model output logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        return_all: If True, return a dict with all metrics; otherwise return just accuracy
        num_classes: Number of classes (excluding -1 which is ignored)
    
    Returns:
        Either a single accuracy value (float) or a dictionary of metrics
    """
    # Filter out labels = -1, and corresponding outputs
    valid_mask = labels != -1
    outputs = outputs[valid_mask]
    labels = labels[valid_mask]
    
    if labels.size(0) == 0:
        # No valid samples
        if return_all:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'balanced_accuracy': 0.0,
                'confusion': np.zeros((num_classes, num_classes))
            }
        return 0.0
    
    # Convert to numpy for sklearn metrics
    # For multi-class, we need predicted class indices
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Calculate basic metrics
    accuracy = accuracy_score(labels_np, preds)
    
    # If requested, calculate additional metrics
    if return_all:
        # Calculate metrics for multi-class classification
        precision = precision_score(labels_np, preds, average='macro', zero_division=0, labels=list(range(num_classes)))
        recall = recall_score(labels_np, preds, average='macro', zero_division=0, labels=list(range(num_classes)))
        f1 = f1_score(labels_np, preds, average='macro', zero_division=0, labels=list(range(num_classes)))
        balanced_acc = balanced_accuracy_score(labels_np, preds)
        
        # Calculate confusion matrix for all classes
        conf_matrix = confusion_matrix(labels_np, preds, labels=list(range(num_classes)))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'confusion': conf_matrix
        }
    
    return accuracy

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

        # Define loss function with optional class weighting
        use_weighted_loss = self.training_config.get('use_weighted_loss', False)
        num_classes = 5  # Classes 0, 1, 2, 3, 4
        
        # Log the class distribution
        if self.verbose:
            total_valid = sum(self.class_counts[i] for i in range(num_classes))
            print("Class distribution:")
            for i in range(num_classes):
                if total_valid > 0:
                    print(f"  Class {i}: {self.class_counts[i]} ({self.class_counts[i]/total_valid*100:.2f}% of valid labels)")
                else:
                    print(f"  Class {i}: {self.class_counts[i]} (0.00% - no valid samples)")
            print(f"  Ignored (-1): {self.class_counts[-1]}")
            
        if use_weighted_loss:
            # Calculate class weights using our helper method
            class_weights = self._calculate_class_weights(self.class_counts, num_classes)
            
            if self.verbose:
                weight_str = ", ".join([f"class {i}: {class_weights[i]:.4f}" for i in range(num_classes)])
                print(f"Using weighted loss with weights: {weight_str}")
            
            # Use CrossEntropyLoss with weights for multi-class classification
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=-1).to(self.device)
        else:
            # Standard CrossEntropyLoss with ignore_index for -1 labels
            self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=-1).to(self.device)

    def _init_data(self):
        self.training_dataset = []
        self.testing_dataset = []
        feature_path = self.data_config["feature_path"]
        feature_path_test = self.data_config["test_feature_path"]
        training_files = glob.glob(os.path.join(feature_path, "*.npz"))
        # training_files = training_files[:2]
        for training_file in tqdm(training_files, desc="Loading training files"):
            new_dataset = ChannelDataset(training_file,  flip=self.training_config["flip"])
            self.training_dataset.append(new_dataset)
        self.training_dataset = ConcatDataset(self.training_dataset)
        if self.verbose: print(f"Total training samples: {len(self.training_dataset)}")
        
        testing_files = glob.glob(os.path.join(feature_path_test, "*.npz"))
        # testing_files = testing_files[:2]
        for testing_file in tqdm(testing_files, desc="Loading testing files"):
            new_dataset = ChannelDataset(testing_file, flip=False)
            self.testing_dataset.append(new_dataset)
        self.testing_dataset = ConcatDataset(self.testing_dataset)
        if self.verbose: print(f"Total testing samples: {len(self.testing_dataset)}")
        
        # Compute class distribution for the training dataset
        if self.verbose: print("Computing class distribution for training data...")
        self.class_counts = self._compute_class_distribution(self.training_dataset)
        if self.verbose: 
             print("Computed class distribution")

    def _compute_class_distribution(self, dataset_or_loader, num_classes=5):
        """
        Compute the distribution of classes in the dataset or dataloader.
        
        Args:
            dataset_or_loader: A Dataset object or DataLoader
            num_classes: Number of classes to count (default: 5 for classes 0-4)
            
        Returns:
            Dictionary with class counts {0: count_0, 1: count_1, ..., -1: count_ignored}
        """
        # Initialize counters for each class (0-4 + -1)
        label_counts = {i: 0 for i in range(num_classes)}  # 0, 1, 2, 3, 4
        label_counts[-1] = 0  # Add special counter for ignored label
        
        # Create loader if passed a dataset
        if not isinstance(dataset_or_loader, DataLoader):
            batch_size = min(1000, len(dataset_or_loader))
            loader = DataLoader(
                dataset_or_loader, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=self.training_config.get("num_workers", 0),
                collate_fn=collate_events
            )
        else:
            loader = dataset_or_loader
        
        # Count labels in each batch
        for batch in tqdm(loader, desc="Counting class distribution", disable=not self.verbose):
            labels = batch['labels']
            
            # Count each class separately
            for i in range(num_classes):  # For classes 0-4
                label_counts[i] += (labels == i).sum().item()
            
            # Count ignored labels (-1)
            label_counts[-1] += (labels == -1).sum().item()
        
        return label_counts
        
    def _calculate_class_weights(self, class_counts, num_classes=5):
        """
        Calculate weights for each class based on their frequency.
        
        Args:
            class_counts: Dictionary with class counts {0: count_0, 1: count_1, ..., -1: count_ignored}
            num_classes: Number of classes (default: 5)
            
        Returns:
            Tensor of class weights
        """
        # Calculate class weights based on computed class distribution
        total_samples = sum(class_counts[i] for i in range(num_classes))
        
        # Calculate weights inversely proportional to class frequency
        class_weights = torch.zeros(num_classes, device=self.device)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (num_classes * class_counts[i])
            else:
                class_weights[i] = 0.0
        
        # Normalize weights so they sum to 1
        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.sum()
            
        return class_weights

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
        num_classes = 5  # Classes 0, 1, 2, 3, 4
        
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
        
        # Handle class imbalance with weighted sampling
        use_weighted_sampling = self.training_config.get("use_weighted_sampling", False)
        if use_weighted_sampling:
            if self.verbose: print("Setting up weighted sampling for class balance")
            
            # Calculate weights for each class using helper method
            class_weights = self._calculate_class_weights(self.class_counts, num_classes)
            
            if self.verbose: 
                weight_str = ", ".join([f"class {i}: {class_weights[i]:.4f}" for i in range(num_classes)])
                print(f"Class weights for sampling: {weight_str}")
            
            # Extract all labels from training subset
            all_labels = []
            for idx in range(len(train_subset)):
                sample = train_subset[idx]
                if isinstance(sample, dict):
                    label = sample['labels']
                else:
                    # Assuming the dataset's __getitem__ returns a dict
                    label = train_subset.dataset[train_subset.indices[idx]]['labels']
                all_labels.append(label)
                
            all_labels = torch.tensor(all_labels)
            
            # Assign weight to each sample based on its class
            # Skip -1 labels (they should be ignored in sampling)
            samples_weight = torch.zeros(len(all_labels))
            for i in range(len(all_labels)):
                label = all_labels[i].long()
                if label >= 0 and label < num_classes:  # Valid class label
                    samples_weight[i] = class_weights[label]
                # Weights for -1 labels remain 0
            
            # Create weighted sampler (if there are any valid samples with weight > 0)
            if samples_weight.sum() > 0:
                sampler = WeightedRandomSampler(
                    weights=samples_weight,
                    num_samples=len(samples_weight),
                    replacement=True
                )
                
                self.train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    sampler=sampler,  # Use the weighted sampler
                    num_workers=num_workers,
                    collate_fn=collate_events,
                    pin_memory=True if 'cuda' in self.device else False,
                )
            else:
                if self.verbose:
                    print("Warning: No valid samples found for weighted sampling. Using regular sampling.")
                self.train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_events,
                    pin_memory=True if 'cuda' in self.device else False,
                )
        else:
            self.train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_events,
                pin_memory=True if 'cuda' in self.device else False,
            )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_events,
            pin_memory=True if 'cuda' in self.device else False
        )

        
        self.test_loader = DataLoader(
            self.testing_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_events,
            pin_memory=True if 'cuda' in self.device else False
        )
        
        if self.verbose: print("DataLoaders created.")
        # Calculate and display label distribution
        # self.compute_label_distribution()
        
    def compute_label_distribution(self):
        """Compute and display the distribution of labels in training and testing datasets."""
        if self.verbose: print("\n--- Label Distribution Analysis ---")
        
        num_classes = 5  # Number of classes to analyze
        
        # Create dataloaders for analysis
        train_all_loader = DataLoader(
            self.training_dataset,
            batch_size=self.training_config["batch_size"],
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 0),
            collate_fn=collate_events
        )
        
        # Use reusable method to compute class distributions
        train_all_counts = self._compute_class_distribution(train_all_loader, num_classes)
        train_only_counts = self._compute_class_distribution(self.train_loader, num_classes)
        val_counts = self._compute_class_distribution(self.val_loader, num_classes)
        test_counts = self._compute_class_distribution(self.test_loader, num_classes)
        
        # Calculate totals
        total_train_all = sum(train_all_counts.values())
        total_valid_train_all = sum(train_all_counts[i] for i in range(num_classes))
        
        total_train_only = sum(train_only_counts.values())
        total_valid_train_only = sum(train_only_counts[i] for i in range(num_classes))
        
        total_val = sum(val_counts.values())
        total_valid_val = sum(val_counts[i] for i in range(num_classes))
        
        total_test = sum(test_counts.values())
        total_valid_test = sum(test_counts[i] for i in range(num_classes))
        
        # Helper function to print distribution for a dataset
        def print_distribution(name, counts, total_valid, total):
            print(f"\n{name}:")
            for i in range(num_classes):
                if total_valid > 0:
                    percentage = counts[i]/total_valid*100
                    print(f"  Class {i}: {counts[i]} ({percentage:.2f}% of valid labels)")
                else:
                    print(f"  Class {i}: {counts[i]} (0.00% - no valid labels)")
                    
            if counts[-1] > 0:
                print(f"  Ignored (-1): {counts[-1]} ({counts[-1]/total*100:.2f}% of all labels)")
            print(f"  Total valid samples: {total_valid}")
            print(f"  Total samples: {total}")
        
        # Display results for all datasets
        print_distribution("All Training Data (before train/val split)", 
                          train_all_counts, total_valid_train_all, total_train_all)
        
        print_distribution("Training Set (after split)", 
                          train_only_counts, total_valid_train_only, total_train_only)
        
        print_distribution("Validation Set", 
                          val_counts, total_valid_val, total_val)
        
        print_distribution("Testing Dataset", 
                          test_counts, total_valid_test, total_test)
        
        # Calculate and display class distribution ratios
        if self.verbose:
            print("\nClass Distribution Ratios (relative to class 0):")
            
            def print_ratios(name, counts):
                if counts[0] > 0:
                    ratios = [counts[i]/counts[0] for i in range(num_classes)]
                    ratio_str = ":".join([f"{r:.2f}" for r in ratios])
                    print(f"  {name}: {ratio_str} (class 0:1:2:3:4)")
                else:
                    print(f"  {name}: N/A (no samples in class 0)")
            
            print_ratios("All Training Data", train_all_counts)
            print_ratios("Training Set", train_only_counts)
            print_ratios("Validation Set", val_counts)
            print_ratios("Testing Set", test_counts)
            
        print("--------------------------------------\n")
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
            waveforms = batch['data'].to(self.device)
            # For CrossEntropyLoss, labels should be of type Long
            labels = batch['labels'].to(self.device).long() # Labels should be [batch_size]

            # Apply preprocessing
            # Ensure preprocessing handles potential extra dimensions if needed
            processed_waveforms = self.pre_processing(waveforms)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass - outputs should be [batch_size, num_classes]
            outputs = self.model(processed_waveforms) # Expected shape [batch_size, num_classes]

            # CrossEntropyLoss expects raw logits (not squeezed)
            # The ignore_index in loss definition takes care of -1 labels
            loss = self.criterion(outputs, labels)
            
            # Reduce the loss before backprop (ignoring -1 labels)
            loss_reduced = loss.mean()  # Manually reduce to a scalar
            loss_reduced.backward()  # Now backward() works on a scalar
            
            self.optimizer.step()

            # Calculate metrics - update to handle multi-class
            batch_loss = loss_reduced.item()  # Get mean loss for reporting
            batch_accuracy = calculate_metrics(outputs, labels)

            total_loss += batch_loss
            total_accuracy += batch_accuracy
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

            # Update tqdm description
            train_iterator.set_postfix(loss=batch_loss, acc=batch_accuracy)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        # calculate accuracy
        avg_accuracy = calculate_metrics(all_outputs, all_labels)
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
        all_outputs = []
        all_labels = []
        all_metadata = [] # To store metadata if needed

        num_batches = len(dataloader)

        # Use tqdm for progress bar
        eval_iterator = tqdm(dataloader, desc="Evaluating", leave=False, disable=not self.verbose)

        with torch.no_grad():
            for batch in eval_iterator:
                waveforms = batch['data'].to(self.device)
                # Convert labels to long for CrossEntropyLoss
                labels = batch['labels'].to(self.device).long()

                # Apply preprocessing
                processed_waveforms = self.pre_processing(waveforms)

                # Forward pass - now expects [batch_size, num_classes]
                outputs = self.model(processed_waveforms)

                # CrossEntropyLoss handles -1 labels via ignore_index
                loss = self.criterion(outputs, labels)
                
                # Calculate mean loss for this batch (excluding -1 labels)
                batch_loss = loss.mean().item()
                total_loss += batch_loss

                # Store outputs, labels, and metadata for potential later analysis
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                all_metadata.append(batch['metadata']) # Store metadata dict

                # Calculate batch metrics for display
                batch_metrics = calculate_metrics(outputs, labels, return_all=True)
                eval_iterator.set_postfix(
                    loss=batch_loss, 
                    acc=batch_metrics['accuracy'], 
                    b_acc=batch_metrics['balanced_accuracy'],
                    f1=batch_metrics['f1']
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate results from all batches
        all_outputs = torch.cat(all_outputs, dim=0) if all_outputs else torch.empty(0)
        all_labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0)
        
        # Calculate overall metrics
        metrics = calculate_metrics(all_outputs, all_labels, return_all=True)
        
        # Combine metadata (this part might need adjustment based on how you want to use it)
        combined_metadata = {k: [item for sublist in [m[k] for m in all_metadata] for item in sublist]
                             for k in all_metadata[0].keys()} if all_metadata else {}
                             
        # For multi-class, predictions are class indices (argmax)
        pred_values = torch.argmax(all_outputs, dim=1).numpy()
        true_values = all_labels.numpy()
        
        combined_metadata[f"channel_pred"] = pred_values
        combined_metadata[f"channel_true"] = true_values

        return avg_loss, metrics, all_outputs, all_labels, combined_metadata

    def train(self):
        """Main training loop."""
        since = time.time()
        best_loss = float('inf')
        best_f1 = -float('inf')
        best_model_path = None
        output_dir = self.data_config.get("output_dir", ".") # Use output_dir from config
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        continue_train = False
        if continue_train:
            # the first 10 epoch was trained with MAML, second order 
            # checkpoint_path = os.path.join(output_dir, "best_model.pth")
            checkpoint_path = '/mnt/SSD1/nipsdataset/anatomical_model/anatomical_model_train_output/cnn2/2025-05-13/run1/best_model.pth'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0

        num_epochs = self.training_config.get("num_epochs", 10) # Get num_epochs from config

        if self.verbose:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Output directory: {output_dir}")

        progress_bar = tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs)
        for epoch in progress_bar:
            # Train one epoch
            train_loss, train_acc, _, _ = self._train_epoch()

            # Validate
            val_loss, val_metrics, _, _, _ = self._evaluate(self.val_loader) # Ignore outputs/labels/metadata for now

            if self.verbose:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")
                print(f"Val Confusion: {val_metrics['confusion']}")
                print("-" * 20)

            # Save the best model based on validation loss
            # if self.val_loader and val_loss < best_loss:
            if self.val_loader and val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
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
        test_loss, test_metrics, test_outputs, test_labels, test_metadata = self._evaluate(self.test_loader)

        if self.verbose:
            num_classes = 5  # Number of classes (0, 1, 2, 3, 4)
            print("\n====== Test Results ======")
            print(f"Loss: {test_loss:.4f}")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            print(f"Macro Precision: {test_metrics['precision']:.4f}")
            print(f"Macro Recall: {test_metrics['recall']:.4f}")
            print(f"Macro F1 Score: {test_metrics['f1']:.4f}")
            
            # Print confusion matrix with labels for multi-class
            conf_matrix = test_metrics['confusion']
            print("\nConfusion Matrix:")
            
            # Print header
            print("          |", end="")
            for i in range(num_classes):
                print(f"  Pred {i}  |", end="")
            print("\n----------|" + "----------|" * num_classes)
            
            # Print each row
            for i in range(num_classes):
                print(f"True {i}   |", end="")
                for j in range(num_classes):
                    print(f"  {conf_matrix[i, j]:<6}  |", end="")
                print()
            
            print("\n==========================")

        # Create a separate metrics dictionary
        metrics_dict = {
            'loss': test_loss,
            'accuracy': test_metrics['accuracy'],
            'balanced_accuracy': test_metrics['balanced_accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'confusion_matrix': test_metrics['confusion'].tolist()
        }

        # Save metrics to a separate JSON file for easier reference
        metrics_path = os.path.join(self.data_config.get("output_dir", "."), "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        # Save test_metadata to a csv file (without nested metrics)
        test_metadata_path = os.path.join(self.data_config.get("output_dir", "."), "test_metadata.csv")
        pd.DataFrame(test_metadata).to_csv(test_metadata_path, index=False)

        return test_loss, test_metrics['balanced_accuracy'], test_outputs, test_labels, test_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["cnn", "ast", "seegnet", "clap"], help="Model name (cnn, ast, seegnet, clap)")
    parser.add_argument("--device", type=str, required=True, help="Device")
    parser.add_argument("--base_output_dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--feature_path", type=str, required=True, help="Path to the training features, generated by omni_ieeg.channel_model.channel_model_train.features.py")
    parser.add_argument("--test_feature_path", type=str, required=True, help="Path to the testing features, generated by omni_ieeg.channel_model.channel_model_train.features_inference.py")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--flip", type=bool, default=True, help="Flip the waveforms for data augmentation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--validation_split_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use_weighted_sampling", type=bool, default=True, help="Use weighted sampling during training")
    parser.add_argument("--use_weighted_loss", type=bool, default=False, help="Use weighted loss function")
    args = parser.parse_args()
    model_name = args.model_name
    device = args.device
    base_output_dir = args.base_output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create date-based subfolder
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
    model_config = model_preprocessing_configs[model_name]["model_config"]
    pre_processing_config = model_preprocessing_configs[model_name]["preprocessing_config"]
    
    data_config = {
        "feature_path": args.feature_path,
        "test_feature_path": args.test_feature_path,
        "output_dir": output_dir,
    }
    training_config = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'device': device,
        'verbose': args.verbose,
        "flip": args.flip,
        "num_workers": args.num_workers,
        "validation_split_ratio": args.validation_split_ratio,
        # Class balancing options - choose one or both
        "use_weighted_sampling": args.use_weighted_sampling,  # Use weighted sampling during training
        "use_weighted_loss": args.use_weighted_loss,     # Use weighted loss function
    }
    
    # Initialize trainer
    trainer = Trainer(data_config, model_config, pre_processing_config, training_config)
    
    # Save the actual class distribution to the config files
    for i in range(5):
        data_config[f"class_{i}_count"] = trainer.class_counts[i]
    data_config["ignored_count"] = trainer.class_counts[-1]
    
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
    
    # --- Start Training ---
    best_model_file = trainer.train()

    # --- Run Testing ---
    if best_model_file:
         print("\n--- Testing Best Model ---")
         trainer.test(model_path=best_model_file)
    else:
         print("\nSkipping testing as no best model was saved.")


     