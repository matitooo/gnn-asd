import argparse
from train import train
from utils import params_extraction

def configuration():
    config_path= "config.yaml"
    params = params_extraction(config_path)
    print("Default parameters for GCN model")
    print (f"Hidden size : {params['hidden']}")
    print (f"Dropout rate : {params['dropout']}")
    print (f"Maximum Polynomial Degree : {params['max_degree']}")
    print ("Default parameters for training")
    print (f"Learning Rate: {params['lrate']}")
    print (f"Weight Decay : {params['decay']}")
    print (f"Training Epochs : {params['epochs']}")
    print (f"Random Seed : {params['seed']}")
    print ("Default Parameters for Graph Creation")
    print (f"Connection Threshold : {params['threshold']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")
    parser.add_argument('--train', action='store_true', help="Train and compare models")
    parser.add_argument('--configuration', action='store_true', help="Print Model configuration")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.configuration:
        configuration()
