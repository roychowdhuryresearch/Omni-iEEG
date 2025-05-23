from omni_ieeg.exploratory_model.anatomical.model_3label.cnn import NeuralCNN, NeuralCNNPreProcessing
from omni_ieeg.exploratory_model.anatomical.model_3label.clap import CLAPClassifier, NeuralClapPreProcessing
from omni_ieeg.exploratory_model.anatomical.model_3label.ast_model import ASTClassifier, NeuralASTPreProcessing
from omni_ieeg.exploratory_model.anatomical.model_3label.seegnet import SEEGNetBinary, SEEGNetPreProcessing

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
                "event_length": 1 * 60 * 1000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms':  1 * 60 * 1000 // 2,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 0
            }
        }
    },
    "cnn12": {
        "model_config": {
            "model_class": NeuralCNN,
            "model_params": {
                "in_channels": 1,
                "outputs": 12 # For multi-class classification
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralCNNPreProcessing,
             "preprocessing_params": {
                'image_size': 224,
                "event_length": 1 * 60 * 1000,
                "frequency": 1000,
                "freq_range_hz": [10, 300],
                'selected_window_size_ms':  1 * 60 * 1000 // 2,
                'selected_freq_range_hz': [10, 300],
                'random_shift_ms': 0
            }
        }
    },
    "clap": {
        "model_config": {
            "model_class": CLAPClassifier,
            "model_params": {
                "num_class": 5,
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
                "num_class": 5,
            }
        },
        "preprocessing_config": {
            "preprocessing_class": NeuralASTPreProcessing,
            "preprocessing_params": {
                'image_size': 224,
                "event_length": 5 * 60 * 200,
                "frequency": 200,
                "freq_range_hz": [1, 100],
                'selected_window_size_ms': 5 * 60 * 1000 // 2,
                'selected_freq_range_hz': [1, 100],
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
    },


}