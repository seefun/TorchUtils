import random
import os
import torch
import numpy as np

import shutil

def seed_everything(seed=42, deterministic=True):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def backup_folder(source='.', destination='../exp/exp1/src'):
    shutil.copytree(source, destination)  

def backup_file(source='param.py', destination='../exp/exp1/parma.py'):
    shutil.copyfile(source, destination)
