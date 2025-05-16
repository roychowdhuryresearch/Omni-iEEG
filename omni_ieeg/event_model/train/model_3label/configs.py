from omni_ieeg.event_model.train.model_3label.cnn import NeuralCNN, NeuralCNNPreProcessing
from omni_ieeg.event_model.train.model_3label.vit import NeuralViT, NeuralViTPreProcessing
from omni_ieeg.event_model.train.model_3label.transformer import PatchTransformer, DummyPreProcessing
from omni_ieeg.event_model.train.model_3label.lstm import AttentiveBiLSTM
from omni_ieeg.event_model.train.model_3label.timesnet_3label import TimesNet
model_preprocessing_configs = {
    "cnn": {
        "model_config": {
            "model_class": NeuralCNN,
            "model_params": {
                "in_channels": 1,
                "outputs": 3 # For multi-class classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralCNNPreProcessing,
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
    "vit": {
        "model_config": {
            "model_class": NeuralViT,
            "model_params": {
                "in_channels": 1,
                "outputs": 3 # For multi-class classification
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
                "seq_len": 285 * 2,
                "hidden_size": 512,
                "num_layers": 3,
                "dropout": 0.3,
                "n_classes": 3
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
                "n_classes": 3
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
                "seq_len": 285 * 2,
                "d_model": 64,
                "top_k": 8,
                "d_ff": 64 * 2,
                "e_layers": 3,
                "embed": "fixed", # not use
                "freq":10,
                "dropout": 0.2,
                "enc_in": 1,
                "num_class": 3,
                "num_kernels": 3,
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
    }
}