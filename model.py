import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg
import torch.nn as nn

class PopulationGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim,K=3, dropout=0.5):
        """
        Population GCN with Chebyshev polynomials (ChebConv).
        K is the maximum degree of Chebyshev polinomyals.
        """
        super(PopulationGCN, self).__init__()
        self.conv1 = tg.ChebConv(input_dim, hidden_dim, K=K)
        self.conv2 = tg.ChebConv(hidden_dim, hidden_dim, K=K)
        self.fc = nn.Linear(hidden_dim,out_dim)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        """
        Takes data object as input and extracts features and edges from it. 
        Returns raw probabilities.
        """
        x, edge_index, edge_weight = data.x.to(self.device), data.edge_index.to(self.device), data.edge_weight.to(self.device)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


class BaselineCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=2, dropout=0.5):
        """
        Two layer 1D CNN. Kernel size and padding must be adjusted layer by layer.
        """
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.fc = nn.Linear(input_dim * hidden_dim, out_dim)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        """
        Takes data object as input and extracts features from it. 
        Returns raw probabilities.
        """
        x = data.x.to(self.device) 
        x = x.unsqueeze(1)  
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(x.size(0), -1) # Flatten vector
        x = self.fc(x)
        return x


