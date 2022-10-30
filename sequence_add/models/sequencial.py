import tensorflow as tf

def create_model(n_cells=100, n_units=1):
    """Creates an LSTM Sequence model according to given parameters

    Args:
        n_cells (int, optional): Number of cells on LSTM. Defaults to 100.
        n_units (int, optional): Number of units on output Dense layer. 
        Defaults to 1.

    Returns:
        object: tf model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(n_cells))
    model.add(tf.keras.layers.Dense(n_units))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, clipvalue=10),
              loss=tf.keras.losses.MeanSquaredError())
    return model

