import torch
import numpy as np
import random
import os

# Device Configuration
if torch.cuda.is_available():
    # Use CUDA device 2 (GPU 7) if it was hardcoded, or just default to cuda:0 if not specified.
    # The original script had cuda:7 hardcoded.
    device = torch.device('cuda:7' if torch.cuda.device_count() > 7 else 'cuda:0')
else:
    device = torch.device('cpu')

# Reproducibility
SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

# Dataset Paths
DATASET_PATH = '/workspace/saumilya/Pratham/pratham/LA/LA'
LABELS_MAP = {'bonafide': 0, 'spoof': 1}
N_MELS = 80
MAX_SECONDS = 2.0

# Ensure the features cache directory exists
FEATURE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'features_cache')
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
