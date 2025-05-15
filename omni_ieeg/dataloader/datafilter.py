import pandas as pd
import os
import glob
from omni_ieeg.dataloader.datasplit import DataSplit

class DataFilter:
    def __init__(self, dataset_root):
        """
        Initialize with dataset root
        
        Parameters:
        - dataset_root: Root directory of the dataset
        """
        self.dataset_root = dataset_root
        # Load participants info once during initialization
        self.participants_df = pd.read_csv(os.path.join(self.dataset_root, "participants.tsv"), sep="\t")
        
    def apply_filters(self, patient_filter=None, edf_filter=None, channel_filter=None):
        """
        Apply filters and return filtered IEEGDataset
        
        Parameters:
        - patient_filter: Function that takes patient row and returns boolean
        - edf_filter: Function that takes edf path and returns boolean
        - channel_filter: Function that takes channel dataframe and returns filtered dataframe
        
        Returns:
        - IEEGDataset with filtered content
        """
        result = DataSplit(dataset_root=self.dataset_root)
        
        # Use the stored participants_df
        for _, row in self.participants_df.iterrows():
            # patient_id = row['participant_id']
            patient_id = str(row['participant_id'])
            dataset_name = row['dataset']
            
            # Apply patient filter
            if patient_filter and not patient_filter(row):
                # print(f"Not passing patient filter: {patient_id}")
                continue
            patient_folder = os.path.join(self.dataset_root, patient_id)
            edf_files = glob.glob(f"{patient_folder}/*/ieeg/*.edf")
            
            for edf_path in edf_files:
                # Apply EDF filter
                if edf_filter and not edf_filter(edf_path):
                    # print(f"Not passing edf filter: {edf_path}")
                    continue
                    
                # Get channels for this EDF
                channels_path = edf_path.replace('_ieeg.edf', '_channels.tsv')
                
                # Skip EDF if channel TSV doesn't exist
                if not os.path.exists(channels_path):
                    # print(f"Not passing channel filter: {channels_path}")
                    continue
                    
                channels_df = pd.read_csv(channels_path, sep="\t")
                
                # Apply channel filter
                if channel_filter:
                    filtered_channels = channel_filter(channels_df.copy())
                    if filtered_channels.empty:
                        # print(f"Not passing channel filter: {channels_path}")
                        continue
                    channels_df = filtered_channels
                    # if nothing left, continue
                    if channels_df.empty:
                        # print(f"Not passing channel filter: {channels_path}")
                        continue
                
                # Add to result
                result.add_edf(dataset_name, patient_id, edf_path, channels_df)
                
        return result
    
    def get_patient_info(self, patient_id):
        """
        Get information for a specific patient from participants.tsv
        
        Parameters:
        - patient_id: The ID of the patient to look up
        
        Returns:
        - Dictionary containing patient information or None if not found
        """
        # Find the patient in the stored participants_df
        patient_row = self.participants_df[self.participants_df['participant_id'] == patient_id]
        
        if patient_row.empty:
            print(f"Warning: Patient {patient_id} not found in participants.tsv")
            return None
            
        # Convert to dictionary
        return patient_row.iloc[0].to_dict()
    
    # has outcome patient filter
    @staticmethod
    def has_outcome_filter():
        """Filter to include only patients with outcome info"""
        return lambda row: row['outcome'] != -1
    
    # Helper methods for common filter patterns
    @staticmethod
    def non_ictal_filter():
        """Filter to include only non-ictal EDFs"""
        return lambda edf_path: 'task-ictal' not in edf_path or ('ictal' in edf_path and 'interictal' in edf_path)
    
    @staticmethod
    def has_soz_filter():
        """Filter to include only channels with SOZ info"""
        return lambda channels_df: channels_df[channels_df['soz'] == 1]
    
    @staticmethod
    def has_resection_filter():
        """Filter to include only channels with resection info"""
        return lambda channels_df: channels_df[channels_df['resection'] == 1]
    @staticmethod
    # Check for EDFs with both SOZ and resection information
    def both_soz_and_resection_filter():
        """Filter to include only channels with both SOZ and resection info"""
        # return channels_df[(channels_df['soz'] != -1) & (channels_df['resection'] != -1)]
        return lambda channels_df: channels_df[(channels_df['soz'] == 1) & (channels_df['resection'] == 1)]
    
    
    @staticmethod
    def has_soz_1000_filter():
        """Filter to include only channels with SOZ info"""
        return lambda channels_df: channels_df[(channels_df['soz'] == 1) & (channels_df['sampling_frequency'] > 990)]
    
    @staticmethod
    def has_resection_1000_filter():
        """Filter to include only channels with resection info"""
        return lambda channels_df: channels_df[(channels_df['resection'] == 1) & (channels_df['sampling_frequency'] > 990)]
    @staticmethod
    # Check for EDFs with both SOZ and resection information
    def both_soz_and_resection_1000_filter():
        """Filter to include only channels with both SOZ and resection info"""
        # return channels_df[(channels_df['soz'] != -1) & (channels_df['resection'] != -1)]
        return lambda channels_df: channels_df[(channels_df['soz'] == 1) & (channels_df['resection'] == 1) & (channels_df['sampling_frequency'] > 990)]
    @staticmethod
    def has_annotation_filter():
        return lambda row: row['event_annotation'] == 1