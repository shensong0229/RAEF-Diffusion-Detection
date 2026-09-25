# AUC、F1、Precision、Recall
# core/metrics/eval_metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    # y_true∈{0,1}, y_score=logits or prob
    scores = sigmoid(y_score)
    y_true = np.asarray(y_true).astype(int)
    # sort by descending score
    order = np.argsort(-scores)
    scores, y_true = scores[order], y_true[order]
    P = (y_true==1).sum()
    N = (y_true==0).sum()
    if P==0 or N==0: return 0.0
    tp=fp=0
    best_tpr=0.0
    for s,t in zip(scores, y_true):
        if t==1: tp+=1
        else: fp+=1
        fpr = fp / N
        tpr = tp / P
        if fpr <= target_fpr:
            best_tpr = tpr
        else:
            break
    return best_tpr

def compute_metrics(y_true, logits):
    probs = sigmoid(logits)
    y_pred = (probs >= 0.5).astype(int)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["roc_auc"] = 0.0
    out["f1"] = float(f1_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred))
    out["recall"] = float(recall_score(y_true, y_pred))
    out["tpr_at_fpr_1e-2"] = float(tpr_at_fpr(y_true, logits, 0.01))
    return out
