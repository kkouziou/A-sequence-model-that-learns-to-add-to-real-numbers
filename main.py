from sequence_add.models import create_model
from sequence_add.processes import train_model, load_model
from datasets import create_xy
from utils import yaml_to_dict, save_artifacts, set_seeds, save_results

import argparse

###################################
## Command line arguments parser ##
###################################
def arguments_parser():
    """Functions for passing command line arguments to main

    Returns:
        object with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=str, 
                        default='configs/config.yaml')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = arguments_parser() # load command line arguments

    params = yaml_to_dict(args.config) # get params from config yaml
    if args.mode == 'train':
        set_seeds(0) 
        x, y = create_xy(params['dataset']['dt_length'], 
                         params['dataset']['t'])
        model = create_model()   
        history = train_model(model, x, y, 
                              params['train']['batch'], 
                              params['train']['epochs'], 
                              params['train']['es_patience'])
        save_artifacts(history, model, params)
    elif args.mode == 'validate':
        set_seeds(43) # different seed from training
        x, y = create_xy(params['dataset']['dt_length'], 
                         params['dataset']['t'])
        model = load_model(f"{params['artifacts']}/model/")
        y_hat = model.predict(x)
        save_results(x, y, y_hat, f"{params['artifacts']}/pred.pkl")
        
        