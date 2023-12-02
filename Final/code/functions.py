# import torch
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from math import sqrt
import time
import warnings
import re
from numpy.random import normal

# from torch import nn
# import torch
# from torch.nn import functional as F
# from torch.optim.lr_scheduler import StepLR
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset

warnings.filterwarnings(action="ignore", category=FutureWarning)



def getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([4] * num, dtype=int)
    rank = pred.argsort()
    output[rank[: int(num * cdf[0] - 1)]] = 0
    output[rank[int(num * cdf[0]) : int(num * cdf[1] - 1)]] = 1
    output[rank[int(num * cdf[1]) : int(num * cdf[2] - 1)]] = 2
    output[rank[int(num * cdf[2]) : int(num * cdf[3] - 1)]] = 3
    if valid:
        cutoff = [pred[rank[int(num * cdf[i] - 1)]] for i in range(4)]
        return output, cutoff
    return output


def getTestScore(pred, cutoff):
    num = pred.shape[0]
    output = np.asarray([4] * num, dtype=int)
    for i in range(num):
        if pred[i] <= cutoff[0]:
            output[i] = 0
        elif pred[i] <= cutoff[1]:
            output[i] = 1
        elif pred[i] <= cutoff[2]:
            output[i] = 2
        elif pred[i] <= cutoff[3]:
            output[i] = 3
    return output


def getTestScore2(pred, cdf):
    num = pred.shape[0]
    rank = pred.argsort()
    output = np.asarray([4] * num, dtype=int)
    output[rank[: int(num * cdf[0] - 1)]] = 0
    output[rank[int(num * cdf[0]) : int(num * cdf[1] - 1)]] = 1
    output[rank[int(num * cdf[1]) : int(num * cdf[2] - 1)]] = 2
    output[rank[int(num * cdf[2]) : int(num * cdf[3] - 1)]] = 3
    return output


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def get_cdf(hist):
    return np.cumsum(hist / np.sum(hist))


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert len(rater_a) == len(rater_b)
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    min_rating, max_rating = None, None
    rater_a, rater_b = np.array(y, dtype=int), np.array(y_pred, dtype=int)
    assert len(rater_a) == len(rater_b)
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator, denominator = 0.0, 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = hist_rater_a[i] * hist_rater_b[j] / num_scored_items
            d = np.square(i - j) / np.square(num_ratings - 1)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


