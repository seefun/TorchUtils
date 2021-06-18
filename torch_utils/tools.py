import random
import os
import torch
import numpy as np

import shutil

def set_gpus(gpus: str):
    ''' 
    setting cuda devices.

    Example:
        >>> set_gpus('0,1,2,3') 
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def seed_everything(seed=42, deterministic=True):
    ''' 
    seed everything (os, np, torch and torch.cuda). 
    deterministic = True : using deterministic algo to make exp reproducible
    deterministic = False: using cudnn.benchmark to speed up training

    Example:
        >>> seed_everything(42) 
    '''
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    """
    used in dataloader to avoid numpy random bug in multi workers pytorch dataloader
    https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    
    Example:
        >>> DataLoader(dataset, batch_size=2, num_workers=4, worker_init_fn=worker_init_fn)
        >>> # And also, in each epoch start, you should do:
        >>> np.random.seed(initial_seed + epoch*999)
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)   

def backup_folder(source='.', destination='../exp/exp1/src'):
    shutil.copytree(source, destination)  

def backup_file(source='param.py', destination='../exp/exp1/parma.py'):
    shutil.copyfile(source, destination)
