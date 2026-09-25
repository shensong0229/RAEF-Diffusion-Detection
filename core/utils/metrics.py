import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

def compute_binary_metrics(y_true, y_prob, thr: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    return {"auc": float(auc), "ap": float(ap), "acc": float(acc), "f1": float(f1)}
