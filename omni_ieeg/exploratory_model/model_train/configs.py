from omni_ieeg.event_model.train.model_3label.cnn import NeuralCNN2, NeuralCNNPreProcessing2
from omni_ieeg.event_model.train.model_3label.cnn_extended import NeuralCNN, NeuralCNNPreProcessing
from omni_ieeg.event_model.train.model_3label.vit import NeuralViT, NeuralViTPreProcessing
from omni_ieeg.event_model.train.model_3label.transformer import PatchTransformer, DummyPreProcessing
from omni_ieeg.event_model.train.model_3label.lstm import AttentiveBiLSTM
from omni_ieeg.event_model.train.model_3label.timesnet_3label import TimesNet, NeuralTimesNetPreProcessing
from omni_ieeg.event_model.train.model_3label.patchtst import PreCompressPatchTST, NeuralPatchTSTPreProcessing

model_preprocessing_configs = {
    "cnn": {
        "model_config": {
            "model_class": NeuralCNN,
            "model_params": {
                "in_channels": 1,
                "outputs": 5 # For multi-class classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralCNNPreProcessing,
             "preprocessing_params": {
                'image_size': 224,
                "event_length": 5 * 60 * 200,
                "frequency": 200,
                "freq_range_hz": [1, 100],
                'selected_window_size_ms':  5 * 60 * 1000 // 2,
                'selected_freq_range_hz': [1, 100],
                'random_shift_ms': 0
            }
        }
    },
    "cnn2": {
        "model_config": {
            "model_class": NeuralCNN2,
            "model_params": {
                "in_channels": 1,
                "outputs": 5 # For multi-class classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralCNNPreProcessing2,
             "preprocessing_params": {
                'image_size': 224,
                "event_length": 5 * 60 * 200,
                "frequency": 200,
                "freq_range_hz": [1, 100],
                'selected_window_size_ms':  5 * 60 * 1000 // 2,
                'selected_freq_range_hz': [1, 100],
                'random_shift_ms': 0
            }
        }
    },
    "vit": {
        "model_config": {
            "model_class": NeuralViT,
            "model_params": {
                "in_channels": 1,
                "outputs": 5 # For multi-class classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralViTPreProcessing,
             "preprocessing_params": {
                'image_size': 224,
                "event_length": 2000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms': 285,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 50
            }
        }
    },
    "lstm": {
        "model_config": {
            "model_class": AttentiveBiLSTM,
            "model_params": {
                "seq_len": 5 * 60 * 200,
                "hidden_size": 256, # 512
                "num_layers": 3,
                "dropout": 0.3,
                "n_classes": 5
            }
        },
        "preprocessing_config": {
            "preprocessing_class": DummyPreProcessing,
            "preprocessing_params": {
                "event_length": 5 * 60 * 200,
                "frequency": 200,
                "selected_window_size_ms": 5 * 60 * 1000 // 2,
                "random_shift_ms": 0
            }
        }
    },
    "transformer": {
        "model_config": {
            "model_class": PatchTransformer,
            "model_params": {
                "seq_len": 285 * 2,
                "patch_len": 30,
                "d_model": 384,
                "n_heads": 8,
                "n_layers": 8,
                "dim_feedforward": 1024,
                "dropout": 0.2,
                "n_classes": 5
            }
        },
        "preprocessing_config": {
            "preprocessing_class": DummyPreProcessing,
            "preprocessing_params": {
                "event_length": 2000,
                "frequency": 1000,
                "selected_window_size_ms": 285,
                "random_shift_ms": 50
            }
        }
    },
    "timesnet": {
        "model_config": {
            "model_class": TimesNet,
            "model_params": {
                "seq_len": 5 * 60 * 200 // 10,
                "d_model": 64,
                "top_k": 8,
                "d_ff": 64 * 2,
                "e_layers": 3,
                "embed": "fixed", # not use
                "freq":10,
                "dropout": 0.2,
                "enc_in": 1,
                "num_class": 5,
                "num_kernels": 3,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralTimesNetPreProcessing,
            "preprocessing_params": {
            }
        }
    },
    "patchtst": {
        "model_config": {
            "model_class": PreCompressPatchTST,
            "model_params": {
                "seq_len": 5 * 60 * 200 // 10,
                "target_length": 1024,
                "num_class": 5,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralPatchTSTPreProcessing,
            "preprocessing_params": {
            }
        }
    },
}