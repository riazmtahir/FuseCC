import os
import random
import numpy as np


def set_global_seed(seed: int):
    """Set global random seeds for reproducibility across runs.

    This sets the PYTHONHASHSEED environment variable and seeds the
    random and numpy RNGs. If PyTorch is installed it will also set
    deterministic seeds for torch CPU/GPU.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
