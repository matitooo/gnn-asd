import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from scipy.spatial import distance
from ABIDE_config import ABIDEParser as Reader
from sklearn.model_selection import train_test_split

def params_extraction(config_path):
    """
    Extracts parameters from config file. Returns a parameter dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    params = {
        'lrate': config['lrate'],
        'epochs': config['epochs'],
        'dropout': config['dropout'],
        'hidden': config['hidden'],
        'decay': config['decay'],
        'max_degree': config['max_degree'],
        'seed': config['seed'],
        'threshold': config['threshold']
    }

    return params

def set_random_seed(seed):   
    """
    Sets random seed for reproducibility
    """       
    np.random.seed(seed)        
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    return None


def data_extraction(subject_IDs, params):
    """
    Extracts features, target and creates graph. Converts graph to a sparse one by building a feature similarity
    matrix and by applying a threshold. Returns X, y, and the final graph in torch format. Labels in the original dataset are 2 for positive and 1 for negative. Here they are converted to 1 and 0.
    """
    

    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    y = np.array([int(labels[sid]) - 1 for sid in subject_IDs], dtype=np.int64) 

    X = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')
    graph = Reader.create_affinity_graph_from_scores(['SEX','SITE_ID'], subject_IDs)

    dist = distance.squareform(distance.pdist(X, metric='correlation'))
    sigma = np.mean(dist)
    sparse_graph = np.exp(-dist**2 / (2 * sigma**2))
    
    final_graph = graph * sparse_graph
    final_graph[final_graph < params['threshold']] = 0

    return torch.from_numpy(X).float(), torch.from_numpy(y).long(), torch.from_numpy(final_graph).float()


def data_object(X, y, edge_index, edge_weight, num_nodes):
    """
    Creates data object to be used in training and evaluation.
    """
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), 
        test_size=0.1, 
        stratify=y,
        random_state=42
    )
    data = Data(
        x=X,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True
    return data
