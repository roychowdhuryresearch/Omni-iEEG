import os
import subprocess
import json

# Define your dataset folder path
dataset_path = '/mnt/SSD1/nipsdataset/release/omniieeg'

# Ensure the dataset-metadata.json file exists
metadata_path = os.path.join(dataset_path, 'dataset-metadata.json')

# Upload the dataset using Kaggle CLI
try:
    subprocess.run(['kaggle', 'datasets', 'create', '-p', dataset_path, '--dir-mode', 'zip'], check=True)
    print("Dataset uploaded successfully.")
except subprocess.CalledProcessError as e:
    print("An error occurred during upload:", e)
