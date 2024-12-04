import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from torch import nn
from spacecutter.models import OrdinalLogisticModel
from skorch import NeuralNet
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss
from sklearn.model_selection import ParameterGrid
import torch
from itertools import combinations


def prob_corr(y_trues, y_preds, quantile = None, return_all = False, possible_labels = None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)

    y_preds = np.pad(y_preds, ((0, 0), (1,1)))
    y_trues_indices += 1

    slice_ = np.array([[0]]) + y_trues_indices
    distribution = y_preds[np.arange(y_preds.shape[0]), slice_].sum(0)

    if quantile is None:
        score = distribution.mean()
    else:
        score = np.quantile(distribution, quantile)
    if return_all:
        return score, distribution
    else:
        return score

def prob_corr_plus_minus(y_trues, y_preds, quantile = None, return_all = False, possible_labels = None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)

    y_preds = np.pad(y_preds, ((0, 0), (1,1)))
    y_trues_indices += 1

    slice_ = np.array([[-1], [0], [1]]) + y_trues_indices
    distribution = y_preds[np.arange(y_preds.shape[0]), slice_].sum(0)

    if quantile is None:
        score = distribution.mean()
    else:
        score = np.quantile(distribution, quantile)
    if return_all:
        return score, distribution
    else:
        return score
    
def accuracy(y_trues, y_preds, possible_labels = None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)
    return (y_trues_indices == y_preds.argmax(1)).sum() / len(y_preds)

def accuracy_plus_minus(y_trues, y_preds, possible_labels = None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)
    return (np.abs(y_trues_indices - y_preds.argmax(1)) <= 1).sum() / len(y_preds)


def average_error(y_trues, y_preds, quantile = None, return_all = False):
    possible_labels = np.unique(y_trues)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)
    distribution = np.abs(y_preds.argmax(-1) - y_trues_indices)
    if quantile is None:
        score = distribution.mean()
    else:
        score = np.quantile(distribution, quantile)
    if return_all:
        return score, distribution
    else:
        return score
    
def bg_score(y_trues, scorer):
    possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)
    s, c = np.unique(y_trues_indices, return_counts = True)
    bg = np.zeros((len(possible_labels[0])))
    bg[s] = c / len(y_trues)
    bg = bg.reshape(1, -1).repeat(len(y_trues), axis = 0)
    return scorer(y_trues, bg)

def fold_change(y_trues, y_preds, scorer, log = False):
    if not log:
        return (1-bg_score(y_trues, scorer)) / (1-scorer(y_trues, y_preds))
    else:
        return np.log2((1-bg_score(y_trues, scorer)) / (1-scorer(y_trues, y_preds)))


def concordance(y_trues, y_preds, possible_labels=None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)

    pred_1D = np.sum(possible_labels * y_preds, 1)

    indices = np.array(list(combinations(np.arange(len(pred_1D)),2)))

    pairs = y_trues_indices[0][indices]
    pairs_label_1, pairs_label_2 = pairs[:, 0], pairs[:, 1]

    pairs_pred = pred_1D[indices]
    pairs_pred_1, pairs_pred_2 = pairs_pred[:, 0], pairs_pred[:, 1]


    label_diff = pairs_label_1 - pairs_label_2
    pred_diff = pairs_pred_1 - pairs_pred_2

    C = (
        np.logical_and(label_diff > 0, pred_diff > 0).sum() +
        np.logical_and(label_diff < 0, pred_diff < 0).sum()
    )
    R = (np.sign(label_diff) == 0).sum() 
    D = (
        np.logical_and(label_diff > 0, pred_diff < 0).sum() +
        np.logical_and(label_diff < 0, pred_diff > 0).sum()
    )
    if (C+D) == 0:
        return 1
    else:
        return (C)/ (C+D)


def concordance_plus_minus(y_trues, y_preds, possible_labels=None):
    if possible_labels is None:
        possible_labels = np.array([2**np.arange(np.log2(np.array(y_trues).min()), np.log2(np.array(y_trues).max()+1))])
    possible_labels = possible_labels.reshape(1, -1)
    y_trues_indices = (np.array([possible_labels]).T == y_trues).argmax(0)

    pred_1D = np.sum(possible_labels * y_preds, 1)

    indices = np.array(list(combinations(np.arange(len(pred_1D)),2)))

    pairs = y_trues_indices[0][indices]
    pairs_label_1, pairs_label_2 = pairs[:, 0], pairs[:, 1]

    pairs_pred = pred_1D[indices]
    pairs_pred_1, pairs_pred_2 = pairs_pred[:, 0], pairs_pred[:, 1]


    label_diff = pairs_label_1 - pairs_label_2
    pred_diff = pairs_pred_1 - pairs_pred_2

    C = (
        np.logical_and(label_diff > 1, pred_diff > 0).sum() +
        np.logical_and(label_diff < -1, pred_diff < 0).sum()
    )

    R = (np.abs(label_diff) <= 1).sum() 

    D = (
        np.logical_and(label_diff > 1, pred_diff < 0).sum() +
        np.logical_and(label_diff < -1, pred_diff > 0).sum()
    )
    if (C+D) == 0:
        return 1
    else:
        return (C)/ (C+D)

class OrdinalModel(BaseEstimator):
    def __init__(self, lr = 1, epochs = 50, l2 = 1e-4, all_labels = [1, 2, 4, 8], **kwargs):
        self.model_params = {"lr" : lr, "max_epochs": epochs, "optimizer__weight_decay" : l2}
        self.labels = all_labels
    def fit(self, X, y):
        labels = self.labels
        self._labels_in_model = labels
        num_classes = len(labels)
        y_indices = (np.array([labels]).T == y.values.reshape(-1)).argmax(0)
        predictor = nn.Sequential(nn.Linear(X.shape[1], 1))
        self.net = NeuralNet(
                module=OrdinalLogisticModel,
                module__predictor=predictor,
                module__num_classes=num_classes,
                criterion=CumulativeLinkLoss,
                train_split=None,
                callbacks=[
                    ('ascension', AscensionCallback()),
                ],
                batch_size = -1,
                verbose = 0,
                device = torch.device("cpu"),
                **self.model_params
            )
        self.net = self.net.fit(X.to_numpy().astype(np.float32), y_indices.reshape(-1, 1))

        self.weights_ = self.net.module_.predictor[0].weight.data[0].cpu().numpy()

        return self

    def predict(self, X):
        preds_ = []
        for x in X.to_numpy():
            predictions = self.net.predict(x.astype(np.float32).reshape(1, -1))
            preds = np.zeros(len(self.labels))
            preds[np.isin(self.labels, self._labels_in_model)] = predictions.reshape(-1)
            preds_.append(preds)
        return np.array(preds_)
    
    def score(self, X, y):
        return concordance_plus_minus(y.to_numpy().reshape(-1), self.predict(X))

    def get_params(self, deep = True):
        return {
            "lr": self.model_params["lr"],
            "epochs": self.model_params["max_epochs"],
            "l2": self.model_params["optimizer__weight_decay"],
            "all_labels": self.labels
        }
    def set_params(self, **params):
        self.model_params = {
            "lr" : params["lr"],
            "max_epochs": params["epochs"],
            "optimizer__weight_decay" : params["l2"]}
        self.labels = params["all_labels"]
        return self


def split_groups_CV(groups, SPLIT_ON, SAMPLE_NAMES, print_ = True):
    sample_codes = pd.Series(groups).str.split('_', expand = True)
    groups_CV = sample_codes[np.where(np.isin(SAMPLE_NAMES, SPLIT_ON))[0]].apply('_'.join, axis = 1).values
    if print_:
        print('Number of groups: ' + str(np.unique(groups_CV).shape[0]))
        for g in np.unique(groups_CV):
            print("%-20s -> %-9s" % (g, groups[g == groups_CV]))

    return groups_CV

def Tune(model, param_grid, X, y, groups, splitter = LeaveOneGroupOut()):
    cv_results = {"params": [], "scores": []}
    grid = ParameterGrid(param_grid)
    for g in grid:
        y_preds = []
        y_trues = []
        y_categories = np.log2(y.astype(float)) - np.log2(y.astype(float).min().item())
        for train_index, test_index in splitter.split(X, y_categories, groups):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = model(**g).fit(X_train, y_train)
            y_preds.append(clf.predict(X_test))
            y_trues.append(y_test.values.reshape(-1))
        score = concordance_plus_minus(np.concatenate(y_trues), np.concatenate(y_preds), possible_labels = param_grid["all_labels"][0])
        cv_results["params"].append(g)
        cv_results["scores"].append(score)
    winning_params = cv_results["params"][np.argmax(cv_results["scores"])]
    model = model(**winning_params).fit(X, y)
    return model, winning_params, cv_results

class UnitarySplitter:
    def __init__(self, n_splits):
        self.cv = StratifiedGroupKFold(n_splits = n_splits)


    def split(self, X, y, groups_CV):
        return [list(self.cv.split(X,y, groups_CV))[0]]