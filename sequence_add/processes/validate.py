import tensorflow as tf

def load_model(filepath):
    """Loads previously saved tf model from filepath

    Args:
        filepath (str): filepath of model folder

    Returns:
        tf model
    """
    model = tf.keras.models.load_model(filepath)
    return model