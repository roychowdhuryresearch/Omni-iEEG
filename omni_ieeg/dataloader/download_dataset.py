#!/usr/bin/env python3
# download dataset from huggingface

import os
import tarfile
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files


def download_and_extract(repo_id, output_dir, delete_tars=True):
    """
    Download files from a Huggingface repo to the specified directory,
    extract all tar files, and optionally delete the original tar files.
    
    Args:
        repo_id (str): The Huggingface repository ID
        output_dir (str): Directory where to save the files
        delete_tars (bool): Whether to delete tar files after extraction
    """
    print(f"Downloading files from '{repo_id}' to {output_dir}")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the repository
    all_files = list_repo_files(repo_id)
    
    # Download each file
    for file_path in tqdm(all_files, desc="Downloading files"):
        try:
            # Download the file
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            
            # Check if it's a tar file that needs extraction
            if local_file.endswith(('.tar', '.tar.gz', '.tgz')):
                extract_dir = os.path.dirname(local_file)
                print(f"Extracting {local_file} to {extract_dir}")
                
                try:
                    with tarfile.open(local_file) as tar:
                        tar.extractall(path=extract_dir)
                    
                    if delete_tars:
                        print(f"Deleting {local_file}")
                        os.remove(local_file)
                except Exception as e:
                    print(f"Error extracting {local_file}: {e}")
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
    
    print("Download and extraction completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract Huggingface datasets")
    parser.add_argument("dataset_name", help="Huggingface dataset name/ID", default="openieeg/ominiieeg")
    parser.add_argument("output_dir", help="Directory to save the dataset", default="/mnt/SSD7/chenda/nips_run/dataset/ominiieeg")
    parser.add_argument("--keep-tars", action="store_true", help="Keep tar files after extraction", default=False)
    
    args = parser.parse_args()
    
    download_and_extract(
        args.dataset_name,
        args.output_dir,
        delete_tars=not args.keep_tars
    )
