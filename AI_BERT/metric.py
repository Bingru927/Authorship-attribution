import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, fbeta_score



def auc(true_y, pred_y):
    n_classes = len(set(pred_y))
    aucs = []
    for i in range(n_classes):
        # the i-th category as a positive example and all other categories as negative examples
        y_true_i = (true_y == i)
        y_pred_i = (pred_y == i)
        auc_i = roc_auc_score(y_true_i, y_pred_i)
        aucs.append(auc_i)
    return aucs

def c_at_1(true_y, pred_y, threshold=0.5):


    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def evaluate_all(true_y, pred_y):

    results = {
                'accuracy': round(accuracy_score(true_y, pred_y), 3),
                'auc': auc(true_y, pred_y),
                'auc_avg': round(sum(auc(true_y, pred_y)) / len(set(true_y)), 3),
               #'c@1': c_at_1(true_y, pred_y),
               'F05': round(fbeta_score(true_y, pred_y, average='weighted',beta=0.5), 3),
               'F1': round(f1_score(true_y, pred_y,average='weighted'), 3)
               }
    results['overall'] = np.mean([results['accuracy'], results['auc_avg'], results['F05'], results['F1']])
    return results


if __name__ == '__main__':
    #true_y = np.asarray([1,0,1,1,1,2,3])
    #pred_y =np.asarray([1,0,1,1,1,2,3])
    true_y = torch.tensor([1,0,1,1,1,2,3])
    pred_y = torch.tensor([1,0,1,1,1,2,3])
    print(evaluate_all(true_y, pred_y))
