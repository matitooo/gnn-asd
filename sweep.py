import torch
from torch.optim import Adam
from torch_geometric.utils import dense_to_sparse
import wandb

from ABIDE_config import ABIDEParser as Reader
from model import PopulationGCN
from utils import params_extraction, data_extraction, data_object, set_random_seed
from model_utils import train_model, eval

def main():
    wandb.init()
    config_path = "config.yaml"
    config = wandb.config

    # Cuda setup and parameters extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = params_extraction(config_path)
    set_random_seed(params['seed'])

    for parameter in params:
        if parameter in wandb.config:  
            params[parameter] = wandb.config[parameter]  


    # Features extraction and Graph Creation
    subject_IDs = Reader.get_ids()
    X, y, A = data_extraction(subject_IDs, params)
    num_nodes = len(subject_IDs)
    num_classes = 2
    edge_index, edge_weight = dense_to_sparse(A)

    # Models Setup
    in_channels = X.shape[1]
    hidden_channels = params['hidden']
    out_channels = num_classes

    # Model and Optimizer Creation
    model_gcn = PopulationGCN(in_channels, hidden_channels, out_channels, dropout=params['dropout']).to(device)
    optimizer_gcn = Adam(model_gcn.parameters(), lr=params['lrate'], weight_decay=params['decay'])

    # Data Object creation
    data = data_object(X, y, edge_index, edge_weight, num_nodes)

    # Training
    print("Training Model: GCN")
    trained_model_gcn = train_model(model_gcn, optimizer_gcn, data, params)

    # Evaluation
    test_auc_gcn, test_f1_gcn, best_threshold_gcn = eval(model_gcn, data)

    print("Final Results")
    print(f"Model: GCN Test AUC: {test_auc_gcn:.4f} Test F1: {test_f1_gcn:.4f} with best threshold: {best_threshold_gcn:.4f}")

    wandb.log({"test_auc": test_auc_gcn, "test_f1": test_f1_gcn})

if __name__ == "__main__":
    main()
