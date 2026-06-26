import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

def compute_metrics(y_true, y_pred, y_probs):
    """
    Computes all clinical metrics required for the research paper.
    y_true/y_pred should be flattened binary arrays.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred), # Sensitivity
        "f1": f1_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "auc_roc": roc_auc_score(y_true, y_probs)
    }
    return metrics