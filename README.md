# Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research

## Overview

Omni-iEEG is a large-scale, comprehensive iEEG dataset and benchmark for epilepsy research. It contains 300+ patients, with over 172 hours of data.

## Getting Started

### Installation

```bash
git clone https://github.com/roychowdhuryresearch/Omni-iEEG.git
cd Omni-iEEG
pip install -e .
```
There is no hard dependency on the python version. We do recommend using python 3.9 or higher to run newer baseline models.

### Data Download

!!! We currently hosting our dataset on huggingface. We are also working on hosting the dataset on other platforms such as OpenNeuro, stay tuned!

Please use the following command to download the dataset from the huggingface dataset. Please note that our full dataset is very large, around 150GB.

```bash
python omni_ieeg/dataloader/download_dataset.py --output_dir /path/to/save/dataset --file sub-openieegDetroit001.tar
```
Please note that if you do not specify the file, this will download the full dataset.
Since the original data is in BIDS format with many edf files, we choose to compress the data and upload. 
The above script will download the compressed data and perform unzipping, which might take a while and require a lot of memory / space.
If you only want to download the metadata, like subject information, here is a list of files you can download:
- `--file sub-openieegDetroit001.tar`: This will download the compressed data for subject Detroit001.
- `--folder derivatives/`: This will download all the files in the derivatives folder, including the HFO detection results, training and testing split information,and the doctor's annotation for HFOs.

### Data format

The unzipped dataset will be in BIDS format, with subfolders begeinning with `sub-` followed by subject id, and `ses-` followed by session id., and then `ieeg` subfolders.
Inside the `ieeg` subfolders, you will see the raw edf files ending with `.edf`, and corresponding channel annotation files ending with `.tsv`.

In the dataset/derivatives folder, we provide three additional subfolders:
- `datasplit`: containing the `final_split.csv` file, which contains the information of the train/val/test split. And `anatomical_mapping.csv` file, which contains the anatomical mapping from raw labels to the proposed labels.

- `hfo`: containing the HFO detection results for each patients, using STE, MNI, and Hilbert detectiong methods. All the results are in 1000Hz timestamps.

- `hfo_annotation`: It contains the doctor's annotation for 30,000+ HFOs events detected by the STE, MNI, and Hilbert methods. We already split them into train and test sets. each parquet file contains the HFOs waveforms, artifact (0 if it is artifact, 1 if it is not), and spike (0 if it is not spike, 1 if it is spike). 

## Usage
### Data loading
We provide helper function to load the dataset, get patient information, perform filtering, etc.
Below is an example of how to use the helper functions to load the dataset.
```python
dataset_folder = "path/to/unzipped/dataset"
data_filter = DataFilter(dataset_folder)

# we want patient that has outcome, edf that is non-ictal, and has both soz and resection channels
filtered_dataset = data_filter.apply_filters(
    patient_filter=DataFilter.has_annotation_filter(),
    edf_filter=DataFilter.non_ictal_filter(),
)
datasets = filtered_dataset.get_datasets() ## get all unique datasets
for dataset in datasets:
    patients = filtered_dataset.get_patients(dataset) # get all patients in the dataset
    test_patient = patients[0] # get the first patient
    edf_files = filtered_dataset.get_edfs_by_patient(test_patient) # get all edf files for the patient
    target_edf = edf_files[0] # get the first edf file
    channel_df = filtered_dataset.get_channel_df(target_edf) # get the channel information and annotationfor the edf file

```

### Event-tasks

#### HFO Detection
We have provided a one-click script to perform HFO detection on the dataset.
```bash
python omni_ieeg/event_model/legacy_model_inference/hfo_detector.py --dataset_path /path/to/unzipped/dataset --resample_freq 1000
```
Please note that this script will perform HFO detection on all the patients and edf files, which might take a while. We choose to resample the data to 1000Hz for easier downstream analysis.

#### Legacy Model (PyHFO, eHFO) classification
Please first use `omni_ieeg/event_model/legacy_model_inference/hfo_features.py` to extract the features. 

Please use `omni_ieeg/event_model/legacy_model_inference/pyhfo_classification.py` to perform the classification using PyHFO checkpoint. Note that we did not provide the checkpoint so you need to download it from the official PyHFO github repo.

Please use `omni_ieeg/event_model/legacy_model_inference/eHFO_classification.py` to perform the classification using eHFO checkpoint. Note that we did not provide the checkpoint so you need to download it from the official eHFO github repo.

#### Training new event classification model
We provide many baseline models in `omni_ieeg/event_model/train/model_3label` folder. You can add your own model by following the same structure, and include that in the `omni_ieeg/event_model/train/model_3label/configs.py`

Then, you can use `omni_ieeg/event_model/train/train_3label.py` to train your model. Note that it requires training and testing features, which is in the `dataset/derivatives/hfo_annotation` folder.

To evaluate the model, you can use `omni_ieeg/event_model/benchmark/benchmark.py` to perform the evaluation.

### Channel and Patient tasks
#### Use event-based model to perform channel-level analysis
Please use `omni_ieeg/channel_model/event_model_inference/inference_3label.py` to perform the classification using event-based model. Note that it use the same features generated by `omni_ieeg/event_model/legacy_model_inference/hfo_features.py`.


#### Training new channel-level model
We provide many baseline models in `omni_ieeg/channel_model/train/model` folder. You can add your own model by following the same structure, and include that in the `omni_ieeg/channel_model/train/model/configs.py`


You need to first generate the features using `omni_eeg\channel_model_train\features.py` and `omni_eeg\channel_model_train\features_infernece.py`. This will automatically extract features from all training and testing set.

Then, you can use `omni_ieeg/channel_model_train/train.py` to train your model using the features you just extracted.

To evaluate the model, you can use `omni_ieeg/channel_model/benchmark/evaluation_channel.py` to perform the evaluation on segment-based models, and use `omni_ieeg/channel_model/benchmark/evaluation_event.py` to perform the evaluation on event-based models.

### Exploratory Tasks
There many potential tasks you can perform on the dataset. We provide script to run anatomical location classification tasks in `omni_ieeg/exploratory_model/anatomical` and ictal/interictal, sleep/awake classification tasks in `omni_ieeg/exploratory_model/ictal_sleep`. It contains very similar structure as in the channel-level model. 
Please note that for anatomical location classification, we have provided the anatomical mapping file in the dataset's derivatives folder. For the ictal/interictal, sleep/awake classification, we have provided the split file in this repo, under `omni_ieeg/exploratory_model/ictal_sleep/`.









