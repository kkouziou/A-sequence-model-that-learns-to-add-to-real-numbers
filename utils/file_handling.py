import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def save_artifacts(history, model, params):
    """Intro point to saving all artifacts

    Args:
        history (dict): containing the train and validation loss
        model (object): tf model
        params (dict): as given in config file
    """
    filepath = params['artifacts']
    if not os.path.exists(filepath): # if no folder exists, create it
        os.makedirs(filepath)
    
    model.save(f'{filepath}/model/')
    save_loss(history, f'{filepath}/loss.pkl')   
    save_loss_plots(history['loss'], history['val_loss'], 
                    f'{filepath}/loss.png') 
    dict_to_yaml(params, f'{filepath}/config_copy.yaml')
    return

def save_loss_plots(loss, val_loss, filepath):
    """Generate and save loss plots vs epoch

    Args:
        loss (array): training loss
        val_loss (array): validation loss
        filepath (str): filepath to save the resulting plot as png
    """
    epochs = np.arange(1,len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.legend(['training loss', 'validation loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train/Validation loss vs epochs')
    plt.savefig(filepath)
    return

def dict_to_yaml(dict_, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(dict_, f)
    return

def yaml_to_dict(filepath):
    with open(filepath) as f:
        dict_ = yaml.load(f, Loader=yaml.FullLoader)
    return dict_

def save_loss(history, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    return

def save_results(x, y, y_hat, filepath):
    """Save the x, y and y_hat arrays as generated from the dataset creation 
    and predicted by the model in a dictionary pickle.

    Args:
        x (array): an array of the input sequences
        y (array): an array of the target values
        y_hat (array): the model's prediction of the target values
        filepath (str): filepath to save the resulting pickle
    """
    result_dict = {'x': x, 'y': y, 'y_hat': y_hat}
    with open(filepath, 'wb') as f:
        pickle.dump(result_dict, f)
    return
    