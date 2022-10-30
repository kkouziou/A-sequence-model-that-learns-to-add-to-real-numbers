import numpy as np

def create_mask(mask_shape):
    """Creates mask to select the two items from the sequence that the 
    model will need to add.

    Args:
        mask_shape (array): the shape of the input sequence as the mask will
        need to have the same shape

    Returns:
        array: mask
    """
    mask = np.zeros(mask_shape)
    mask[:,:2,0] = 1
    rng = np.random.default_rng()
    for row in mask:
        rng.shuffle(row)
    return mask

def create_xy(dataset_length, T):
    """Creates random xy pairs according to uniform distribution

    Args:
        dataset_length (int): length of the dataset
        T (int): length of each sequence

    Returns:
        array: x, y pairs to train the sequence model
    """
    sequence = np.random.rand(dataset_length, T, 1)
    mask = create_mask(sequence.shape)
    
    x = np.concatenate((sequence, mask),-1)
    y = np.sum((x[:,:,0]*x[:,:,1]),axis=-1)
    return x, y
    