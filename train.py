import torch
from torch.optim import Adam
from torch_geometric.utils import dense_to_sparse
from ABIDE_config import ABIDEParser as Reader
from model import PopulationGCN, BaselineCNN
from utils import params_extraction, data_extraction, data_object, set_random_seed
from model_utils import train_model, eval

def train():
    # Cuda setup and parameters extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = "config.yaml"
    params = params_extraction(config_path)
    set_random_seed(params['seed'])

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

    # Models and Optimizers Creation
    model_gcn = PopulationGCN(in_channels, hidden_channels, out_channels, dropout=params['dropout']).to(device)
    optimizer_gcn = Adam(model_gcn.parameters(), lr=params['lrate'], weight_decay=params['decay'])
    model_cnn = BaselineCNN(in_channels, hidden_channels, out_channels, dropout=params['dropout']).to(device) 
    optimizer_cnn = Adam(model_cnn.parameters(), lr=params['lrate'], weight_decay=params['decay'])

    # Data Object creation
    data = data_object(X, y, edge_index, edge_weight, num_nodes)

    # Training
    print("Training Model: GCN")
    trained_model_gcn = train_model(model_gcn, optimizer_gcn, data, params)
    print("Training Model: CNN")
    trained_model_cnn = train_model(model_cnn, optimizer_cnn, data, params)

    # Evaluation
    test_auc_gcn, test_f1_gcn, best_threshold_gcn = eval(model_gcn, data)
    test_auc_cnn, test_f1_cnn, best_threshold_cnn = eval(model_cnn, data)

    print("Final Results")
    print(f"Model: GCN Test AUC: {test_auc_gcn:.4f} Test F1: {test_f1_gcn:.4f} with best threshold: {best_threshold_gcn:.4f}")
    print(f"Model: CNN Test AUC: {test_auc_cnn:.4f} Test F1: {test_f1_cnn:.4f} with best threshold: {best_threshold_cnn:.4f}")
