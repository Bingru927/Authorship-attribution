import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, fbeta_score



def auc(true_y, pred_y):
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


def c_at_1(true_y,scores, threshold=0.5):


    n = float(len(true_y))
    nc, nu = 0.0, 0.0

    for y, p in zip(true_y,scores):
        if p[0] == 0.5:
            nu += 1.0
        elif (p[0]< 0.5) == (y > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def evaluate_all(true_y,scores):
    #print(scores)
    
    pred_y = [0 if p[0] >0.5 else 1 for p in scores ]
    
   
    results = {
                'accuracy':accuracy_score(true_y, pred_y),
                'auc': auc(true_y, pred_y),
               'c@1': c_at_1(true_y, scores),
               'F05': fbeta_score(true_y, pred_y, average='weighted',beta=0.5),
               'F1': f1_score(true_y, pred_y,average='weighted')
               }
    results['overall'] = np.mean(list(results.values()))

    for k, v in results.items():
        results[k] = round(v, 3)

    return results



if __name__ == '__main__':
    true_y = [1,0,1,1]
    scores = [[0,1], [0,1] ,[1,1],[0, 0.5]]
    
    print(evaluate_all(true_y, scores))