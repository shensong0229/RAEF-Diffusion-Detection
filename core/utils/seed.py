import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 追求效果优先：benchmark=True 更快且通常更好
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
