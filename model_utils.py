import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
from tqdm import tqdm 

def train_model(model, optimizer, data, params):
    """
    Trains model using cross entropy loss as loss function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in tqdm(range(params['epochs'])):
        optimizer.zero_grad()
        out = model(data)
        out = out[data.train_mask]
        y_true = data.y[data.train_mask].to(device)
        loss = F.cross_entropy(out, y_true)
        loss.backward()
        optimizer.step()
    return model

def eval(model, data):
    """
    Evaluates model in terms of AUC score and F1 score. The best F1 score is found by varying the classification threshold. Tensors are moved to cpu as sklearn requires them to be on cpu to perform ROC computations.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data)
        preds = logits.argmax(dim=1)
        y_true = data.y[data.test_mask].cpu().numpy() 
        y_score = logits[data.test_mask].softmax(dim=1)[:, 1].cpu().numpy()
        
        test_auc = roc_auc_score(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8) 
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)

    return test_auc, best_f1, best_threshold
