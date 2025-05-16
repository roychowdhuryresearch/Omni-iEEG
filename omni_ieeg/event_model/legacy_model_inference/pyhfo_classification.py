import os, time, copy, sys

import torch
import torch.nn as nn
import torch.optim as optim
from random import random, sample
import numpy as np
from torch.utils.data import dataset, Subset, WeightedRandomSampler

import torch.utils.data as data
import random
import torch
from torch.utils.data import  DataLoader
from sklearn.model_selection import  KFold
import copy

from tqdm import tqdm
from pathlib import Path
import torch
from torchvision.models import resnet18
import pandas as pd
from omni_ieeg.event_model.pyhfo_classification.args import args, feature_param
from omni_ieeg.event_model.pyhfo_classification.model import PreProcessing
from omni_ieeg.event_model.pyhfo_classification.features import generate_feature_from_df_gpu_batch, normalize_img
from omni_ieeg.dataloader.datafilter import DataFilter
import argparse
# Redirect 'src' module to the current module structure
sys.modules['src'] = sys.modules['omni_ieeg.event_model.pyhfo_classification']
print("Redirected 'src' module to 'omni_ieeg.event_model.pyhfo_classification'")

def inference_one_patient(feature_path, model, device, feature_param, preprocessing):
    data = np.load(feature_path, allow_pickle=True)
    start = data["start"]
    end = data["end"]
    real_start = data["real_start"]
    real_end = data["real_end"]
    is_boundary = data["is_boundary"]
    channel_name = data["name"]
    hfo_waveforms = data["hfo_waveforms"]
    detector = data["detector"]
    participant = data["participant"]
    session = data["session"]
    file_name = data["file_name"]
    if len(hfo_waveforms) == 0:
        print(f"No HFO waveforms found in {feature_path}")
        cols = ["start", "end", "real_start", "real_end", "is_boundary", "name", "detector", "participant", "session", "file_name"] + list(model.keys())
        return pd.DataFrame(columns=cols)

    batch_size = 32
    out = []
    for i in range(0, len(hfo_waveforms), batch_size):
        hfo_waveforms_batch = torch.tensor(hfo_waveforms[i:i+batch_size]).to(device)
        start_batch = start[i:i+batch_size]
        end_batch = end[i:i+batch_size]
        real_start_batch = real_start[i:i+batch_size]
        real_end_batch = real_end[i:i+batch_size]
        is_boundary_batch = is_boundary[i:i+batch_size]
        channel_name_batch = channel_name[i:i+batch_size]
        detector_batch = detector[i:i+batch_size]
        participant_batch = participant[i:i+batch_size]
        session_batch = session[i:i+batch_size]
        file_name_batch = file_name[i:i+batch_size]
        output = {}
        with torch.no_grad():
            for processor_name, processor in preprocessing.items():
                hfo_features_batch = generate_feature_from_df_gpu_batch(hfo_waveforms_batch, feature_param, processor_name, device)
                hfo_features_batch = normalize_img(hfo_features_batch)
                hfo_features_batch = processor(hfo_features_batch)
                output[processor_name] = model[processor_name](hfo_features_batch).cpu().numpy()

        for j in range(len(start_batch)):
            result_dict = {}
            result_dict["start"] = start_batch[j]
            result_dict["end"] = end_batch[j]
            result_dict["real_start"] = real_start_batch[j]
            result_dict["real_end"] = real_end_batch[j]
            result_dict["is_boundary"] = is_boundary_batch[j]
            result_dict["name"] = channel_name_batch[j]
            result_dict["detector"] = detector_batch[j]
            result_dict["participant"] = participant_batch[j]
            result_dict["session"] = session_batch[j]
            result_dict["file_name"] = file_name_batch[j]
            for model_name, model_output in output.items():
                result_dict[model_name] = model_output[j]
            out.append(result_dict)
    out_df = pd.DataFrame(out)
    return out_df

def inference_single(feature_dir, feature_path, models, preprocessings, device, args, feature_param):
    """
    Performs inference for a single participant using paths obtained from filtered_dataset.
    Checks for existence of both HFO CSV and Feature NPZ files.

    Args:
        participant_id (str): The ID of the participant.
        filtered_dataset (DataSplit): The filtered dataset object containing EDF paths.
        models (dict): Dictionary of loaded models.
        preprocessings (dict): Dictionary of preprocessing functions.
        device (str): The device to run inference on (e.g., 'cuda:0').
        args (dict): Dictionary of arguments including dataset_dir, feature_dir, output_dir, hfo_dir.
        feature_param (dict): Dictionary of feature parameters.
    """
    # Perform inference using the feature file
    save_path = feature_path.replace(".npz", ".csv")
    save_path = os.path.join(args["output_dir"], save_path)
    if os.path.exists(save_path):
        print(f"Already exists, loading results from {save_path}")
        out_df = pd.read_csv(save_path)
    else:
        out_df = inference_one_patient(os.path.join(feature_dir, feature_path), models, device, feature_param, preprocessings)

        # Save the results
        if out_df is not None and not out_df.empty:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_df.to_csv(save_path, index=False)
            print(f"Saved classification results to {save_path}")
        else:
            print(f"No results generated or empty results for {feature_path}. Not saving file.")
    return out_df

def inference_pipeline(args, device, feature_param):
    models = {}
    preprocessings = {}
    for model_name, model_path in args["model_path"].items():
        print(f"Loading model: {model_name}")
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            models[model_name] = ckpt["model"].to(device).float()
            models[model_name].eval()
            preprocessing_dict = ckpt["preprocessing"]
            preprocessing_dict["fs"] = feature_param["resample"]
            print(f"Preprocessing for {model_name}: {preprocessing_dict}")
            preprocessings[model_name] = PreProcessing.from_dict(preprocessing_dict)
        except KeyError as e:
             print(f"Error loading model {model_name}: Missing key {e} in checkpoint. Ensure 'model' and 'preprocessing' keys exist.")
             if model_name in models: del models[model_name]
             continue
        except Exception as e:
            print(f"Error loading model {model_name} from {model_path}: {e}")
            if model_name in models: del models[model_name]
            continue

    if not models:
        print("No models loaded successfully. Exiting.")
        return
    feature_dir = args["feature_dir"]
    features = os.listdir(feature_dir)
    features = [f for f in features if f.endswith(".npz")]
    # reverse the order of features
    features = features[::-1]
    out_dfs = []
    for feature_path in tqdm(features, desc="Processing features"):
        out_df = inference_single(feature_dir, feature_path, models, preprocessings, device, args, feature_param)
        out_dfs.append(out_df)
    out_dfs = pd.concat(out_dfs)
    out_dfs.to_csv(os.path.join(args["output_dir"], "all_results.csv"), index=False)

    print("Inference pipeline finished, all results saved to ", os.path.join(args["output_dir"], "all_results.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the results")
    parser.add_argument("--feature_dir", type=str, required=True, help="Path to the features, generated by omni_ieeg.event_model.legacy_model_inference.hfo_features.py")
    parser.add_argument("--pyhfo_artifact_pruning_path", type=str, required=True, help="Path to the pyhfo_artifact_pruning model, please refer to pyhfo paper")
    parser.add_argument("--pyhfo_spike_pruning_path", type=str, required=True, help="Path to the pyhfo_spike_pruning model, please refer to pyhfo paper")
    args = parser.parse_args()
    
    
    device = "cuda:1"
    args['device'] = device
    args['output_dir'] = args.output_dir
    os.makedirs(args['output_dir'], exist_ok=True)
    args["feature_dir"] = args.feature_dir
    args["model_path"] = {"pyhfo_artifact_pruning": args.pyhfo_artifact_pruning_path,
                          "pyhfo_spike_pruning": args.pyhfo_spike_pruning_path}
    feature_param["model_additional_parameter"] = {"pyhfo_artifact_pruning": {"n_feature": 1, "time_window_ms": 1000},
                                          "pyhfo_spike_pruning": {"n_feature": 2, "time_window_ms": 1000}}
    feature_param["n_jobs"] = 32
    feature_param["n_feature"] = 1
    feature_param["resample"] = 1000
    feature_param["raw_waveform_length"] = 2000
    feature_param["freq_min_hz"] = 10
    feature_param["freq_max_hz"] = 500
    feature_param["image_size"] = 224
    inference_pipeline(args, device, feature_param)
