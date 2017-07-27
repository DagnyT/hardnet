"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

"""
import numpy as np
def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point. 
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of 
    # 'recall_point' of the total number of elements with label==1. 
    # (np.argmax returns the first occurrence of a '1' in a bool array). 
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels)) 

    FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    return float(FP) / float(FP + TN)
'''import operator


def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores.sort(key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count
'''
