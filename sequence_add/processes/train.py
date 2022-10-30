import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

def train_model(model, x, y, batch_size, epochs, patience=10):
    """Function that trains the model given spesific parameters and inputs

    Args:
        model (obj): tf model
        x (array): model input array with dims [dt_length, T, 2]
        y (array): target values array with dims [dt_length, T, 1]
        batch_size (int): batch size
        epochs (int): max number of epochs
        patience (int, optional): Early stopping patience. Defaults to 10.

    Returns:
        dict: dict keys are loss (training loss) and val_loss
    """
    es = EarlyStopping(monitor='loss', patience=patience, 
                       restore_best_weights=True) # tf early stopping callback
    history = model.fit(x, y, batch_size, epochs, validation_split=0.7, 
                        callbacks=[es]) # 30% of the data are automatically seperated and used for validation
    return history.history
