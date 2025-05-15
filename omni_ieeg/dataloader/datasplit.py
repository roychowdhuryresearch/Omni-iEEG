import pandas as pd
import os
import numpy as np
class DataSplit:
    def __init__(self, dataset_root):
        """
        Represents filtered dataset structure
        {dataset: {patient: [edf_info]}}
        where edf_info contains path and associated channel data
        
        Parameters:
        - dataset_root: Root directory of the dataset (optional)
        """
        self.data = {}  # Structure: {dataset_name: {patient_id: [edf_info_dict]}}
        self.dataset_root = dataset_root
        
        # Load participants info if dataset_root is provided
        self.participants_df = None
        participants_path = os.path.join(self.dataset_root, "participants.tsv")
        if os.path.exists(participants_path):
            self.participants_df = pd.read_csv(participants_path, sep="\t")
        else:
            raise ValueError("participants.tsv file not found in dataset_root")
        
    def add_edf(self, dataset_name, patient_id, edf_path, channels_df=None):
        """Add an EDF file to the structure"""
        if dataset_name not in self.data:
            self.data[dataset_name] = {}
            
        if patient_id not in self.data[dataset_name]:
            self.data[dataset_name][patient_id] = []
            
        edf_info = {
            'path': edf_path,
            'channels': channels_df
        }
        
        self.data[dataset_name][patient_id].append(edf_info)
        
    def get_datasets(self):
        """Get list of available datasets"""
        return list(self.data.keys())
        
    def get_patients(self, dataset=None):
        """Get list of patients, optionally for a specific dataset"""
        if dataset:
            return list(self.data.get(dataset, {}).keys())
            
        all_patients = []
        for dataset_patients in self.data.values():
            all_patients.extend(dataset_patients.keys())
        return all_patients
        
    def get_edfs_by_patient(self, patient_id):
        """Get all EDFs for a specific patient"""
        for dataset_patients in self.data.values():
            if patient_id in dataset_patients:
                return [edf_info['path'] for edf_info in dataset_patients[patient_id]]
        return []
        
    def get_all_edfs(self):
        """Get all EDFs"""
        all_edfs = []
        for dataset_patients in self.data.values():
            for patient_edfs in dataset_patients.values():
                all_edfs.extend([edf_info['path'] for edf_info in patient_edfs])
        return all_edfs
    
    def get_channels_for_edf(self, edf_path):
        """Get channels dataframe for a specific EDF"""
        for dataset_patients in self.data.values():
            for patient_edfs in dataset_patients.values():
                for edf_info in patient_edfs:
                    if edf_info['path'] == edf_path:
                        return edf_info['channels']
        return None
        
    def get_merged_channels_for_patient(self, patient_id):
        """
        Get merged channel dataframes for a patient, ensuring rows with the same name are identical
        
        Parameters:
        - patient_id: The ID of the patient to look up
        
        Returns:
        - Merged dataframe or None if no channels found
        """
        # Get all EDFs for the patient
        edf_paths = self.get_edfs_by_patient(patient_id)
        if not edf_paths:
            return None
            
        # Get all channel dataframes
        channel_dfs = [self.get_channels_for_edf(edf_path) for edf_path in edf_paths]
        channel_dfs = [df for df in channel_dfs if df is not None]
        
        if not channel_dfs:
            return None
            
        # Create a dictionary to store unique channels by name
        unique_channels = {}
        
        # Process each dataframe
        for df in channel_dfs:
            # Iterate through each row in the dataframe
            for _, row in df.iterrows():
                name = row['name']
                
                # If this channel name is already in our dictionary
                if name in unique_channels:
                    # Check if the existing row is identical to this one
                    existing_row = unique_channels[name]
                    
                    # Special case: if one row has a value and the other has -1
                    # We want to keep the row with the actual value
                    if not row.equals(existing_row):
                        # Check if the difference is only in columns with -1 values
                        # or in acceptable columns that can differ between EDFs
                        is_special_case = True
                        for col in row.index:
                            if row[col] != existing_row[col]:
                                # These columns can differ between EDFs, so we ignore differences here
                                if col in ["low_cutoff", "high_cutoff", "sampling_frequency", "edf_length"]:
                                    continue
                                
                                # If both values are not -1 or "-1", this is not our special case
                                val1 = row[col]
                                val2 = existing_row[col]
                                
                                # Check for both string "-1" and integer -1
                                is_val1_minus1 = val1 == -1 or val1 == "-1"
                                is_val2_minus1 = val2 == -1 or val2 == "-1"
                                
                                if not is_val1_minus1 and not is_val2_minus1:
                                    is_special_case = False
                                    break
                        
                        if is_special_case:
                            # Keep the row with the non -1 value
                            merged_row = existing_row.copy()
                            for col in row.index:
                                # Skip these columns as they can differ between EDFs
                                if col in ["low_cutoff", "high_cutoff", "sampling_frequency", "edf_length"]:
                                    continue
                                    
                                val1 = row[col]
                                val2 = existing_row[col]
                                
                                is_val1_minus1 = val1 == -1 or val1 == "-1"
                                is_val2_minus1 = val2 == -1 or val2 == "-1"
                                
                                if not is_val1_minus1 and is_val2_minus1:
                                    # Current row has a value where existing row has -1
                                    merged_row[col] = val1
                            
                            # Update the dictionary with the merged row
                            unique_channels[name] = merged_row
                        else:
                            # If not our special case, print out what is different
                            print(f"Channel {name} has different values across EDFs for patient {patient_id}")
                            print(row)
                            print(existing_row)
                            if name is not np.nan and name is not None:
                                raise ValueError(f"Channel {name} has different values across EDFs for patient {patient_id}")
                else:
                    # If not in dictionary, add it
                    unique_channels[name] = row
        
        # Convert dictionary to dataframe
        if unique_channels:
            return pd.DataFrame(list(unique_channels.values()))
        else:
            return None
            
    def get_patient_per_dataset_count(self):
        """Get patient count per dataset"""
        result = {}
        for dataset_name in self.get_datasets():
            result[dataset_name] = len(self.get_patients(dataset_name))
        return result
    
    def get_edf_per_dataset_count(self):
        """Get EDF count per dataset"""
        result = {}
        for dataset_name in self.get_datasets():
            if dataset_name not in result:
                result[dataset_name] = 0
            for patient_id in self.get_patients(dataset_name):
                result[dataset_name] += len(self.get_edfs_by_patient(patient_id))
        return result

    def get_patient_info(self, patient_id):
        """
        Get information for a specific patient from participants.tsv
        
        Parameters:
        - patient_id: The ID of the patient to look up
        
        Returns:
        - Dictionary containing patient information or None if not found
        """
        if self.participants_df is None:
            raise ValueError("participants_df is not loaded. Make sure dataset_root is set correctly.")
            
        # Find the patient in the stored participants_df
        patient_row = self.participants_df[self.participants_df['participant_id'] == patient_id]
        
        if patient_row.empty:
            print(f"Warning: Patient {patient_id} not found in participants.tsv")
            return None
        
        # Convert to dictionary
        return patient_row.iloc[0].to_dict()
        
    def get_patient_dataset(self, patient_id):
        """
        Get the dataset name for a specific patient
        
        Parameters:
        - patient_id: The ID of the patient to look up
        
        Returns:
        - Dataset name as string or None if not found
        """
        # First try to find in the data structure
        for dataset_name, patients in self.data.items():
            if patient_id in patients:
                return dataset_name
                
        # If not found in data structure, try to find in participants_df
        if self.participants_df is not None:
            patient_row = self.participants_df[self.participants_df['participant_id'] == patient_id]
            if not patient_row.empty:
                return patient_row.iloc[0]['dataset']
                
        print(f"Warning: Patient {patient_id} not found")
        return None
