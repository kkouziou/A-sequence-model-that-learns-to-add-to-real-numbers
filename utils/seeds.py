import os
import random
import tensorflow as tf
import numpy as np

def set_seeds(seed):
    """Set random seeds to ensure reproducable results

    Args:
        seed (int): seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    return