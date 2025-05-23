from omni_ieeg.exploratory_model.ictal_sleep.model.cnn import NeuralCNN, NeuralCNNPreProcessing
from omni_ieeg.exploratory_model.ictal_sleep.model.clap import CLAPClassifier, NeuralClapPreProcessing
from omni_ieeg.exploratory_model.ictal_sleep.model.ast_model import ASTClassifier, NeuralASTPreProcessing
from omni_ieeg.exploratory_model.ictal_sleep.model.seegnet import SEEGNetBinary, SEEGNetPreProcessing

model_preprocessing_configs = {
    "cnn": {
        "model_config": {
            "model_class": NeuralCNN,
            "model_params": {
                "in_channels": 1,
                "outputs": 1 # For binary classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralCNNPreProcessing,
             "preprocessing_params": {
                'image_size': 224,
                "event_length": 60000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms': 30000,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 0
            }
        }
    },
    "lstm": {
        "model_config": {
            "model_class": AttentiveBiLSTM,
            "model_params": {
                "seq_len": 1 * 60 * 1000,
                "hidden_size": 256, # 512
                "num_layers": 3,
                "dropout": 0.3,
                "n_classes": 1
            }
        },
        "preprocessing_config": {
            "preprocessing_class": DummyPreProcessing,
            "preprocessing_params": {
                "event_length": 1 * 60 * 1000,
                "frequency": 1000,
                "selected_window_size_ms": 1 * 60 * 1000 // 2,
                "random_shift_ms": 0
            }
        }
    },
    "vit": {
        "model_config": {
            "model_class": NeuralViT,
            "model_params": {
                "in_channels": 1,
                "outputs": 1
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralViTPreProcessing,
            "preprocessing_params": {
                'image_size': 224,
                "event_length": 1 * 60 * 1000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms': 1 * 60 * 1000 // 2,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 0
            }
        }
    },
    "transformer": {
        "model_config": {
            "model_class": PatchTransformer,
            "model_params": {
                "seq_len": 1 * 60 * 1000,
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
                "random_shift_ms": 0
            }
        }
    },
    "timesnet": {
        "model_config" : {
            "model_class": TimesNet,
            "model_params": {
                "seq_len": 1 * 60 * 300,
                "d_model": 64,
                "top_k": 8,
                "d_ff": 64 * 2,
                "e_layers": 3,
                "embed": "fixed", # not use
                "freq":10,
                "dropout": 0.2,
                "enc_in": 1,
                "num_class": 1,
                "num_kernels": 3,
            }
        },
        "preprocessing_config" : {
            "preprocessing_class": NeuralTimesNetPreProcessing,
            "preprocessing_params": {
            }
        }
    },
    "clap": {
        "model_config": {
            "model_class": CLAPClassifier,
            "model_params": {
                "num_class": 1,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralClapPreProcessing,
            "preprocessing_params": {
            }
        }
    },
    "ast": {
        "model_config": {
            "model_class": ASTClassifier,
            "model_params": {
                "num_class": 1,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralASTPreProcessing,
            "preprocessing_params": {
                'image_size': 224,
                "event_length": 1 * 60 * 1000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms': 1 * 60 * 1000 // 2,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 0
            }
        }
    },
    "seegnet": {
        "model_config": {
            "model_class": SEEGNetBinary,
            "model_params": {
                "in_channels": 1,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": SEEGNetPreProcessing,
            "preprocessing_params": {
            }
        }
    }
}