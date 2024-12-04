import torch
torch.set_num_threads(4)
import argparse
from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
import seaborn as sns
import matplotlib.pyplot as plt
import math

    
def read_data(path):
    dataframe = pd.read_csv(path,index_col=0)
    groups = dataframe.index
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, [-1]]
    return X, y, groups


def weights_plot(weights, X):
    fig, axes = plt.subplots(1, 1, figsize=np.array((5,11)))
    sns.barplot(data = list(weights.T), orient = "h", ax = axes, errorbar="sd")
    axes.set_ylabel('Feature')
    axes.set_xlabel('Weight')
    axes.set_yticklabels(X.columns)
    return fig, axes


def preds_plot(y_trues, y_preds, param_grid, sample_names):
    rows = math.ceil(len(y_trues) / 10)
    cols = 10
    fig, axes = plt.subplots(rows, cols, sharey = True, sharex = True, figsize=np.array((30,6 + 6*rows/5)))

    n_ = y_preds.shape[1]

    palette = sns.color_palette("rocket", n_colors=n_)[::-1]
    c = 0
    for i in range(len(y_trues)):
        barlist = axes[c // 10, c % 10].bar(np.arange(n_), y_preds[i,:])
        
        axes[c // 10, c % 10].set_xticks(np.arange(n_))
        axes[c // 10, c % 10].set_xticklabels(param_grid["all_labels"][0], rotation=45, ha='right')
        axes[c // 10, c % 10].set_ylabel('Probability')
        axes[c // 10, c % 10].set_xlabel('MIC')
        axes[c // 10, c % 10].set_title(sample_names[i] + "\n" + 'True label: ' +  str(y_trues[i]))
        axes[c // 10, c % 10].xaxis.set_tick_params(labelbottom=True)
        axes[c // 10, c % 10].yaxis.set_tick_params(labelleft=True)
        axes[c // 10, c % 10]
        location_true = np.where(param_grid["all_labels"][0] == y_trues[i])[0][0]
        distances_from_true = [np.abs(i - location_true) for i in range(len(param_grid["all_labels"][0]))]
        
        for k in range(len(param_grid["all_labels"][0])):
            barlist[k].set_color(palette[distances_from_true[k]])
        
        c += 1 
    return fig, axes


def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Script for training ordinal regression models on DNA variant data (LOOCV)",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("data_path", type=str, metavar="data_path", help="path to data file")
    parser.add_argument(
        "--save_weights_path_prefix",
        type=str,
        default="",
        help="Prefix path for the file to save weights.csv to. Default \'\', which means don't save"
    )
    parser.add_argument(
        "--save_preds_path_prefix",
        type=str,
        default="",
        help="Prefix path for the file to save preds.csv to. Default \'\', which means don't save"
    )
    parser.add_argument(
        "--save_figs_path_prefix",
        type=str,
        default="",
        help="Prefix path for the file to save weights.csv to. Default \'\', which means don't save"
    )

    args = parser.parse_args()

    X, y, groups = read_data(args.data_path)

    groups_CV = split_groups_CV(
        groups,
        ["Strain", "Lineage no"],
        ["Frozen stock identifier", "Lineage no", "Condition", "Strain"],
        print_ = True
    )

    param_grid = {
        "lr": [1, 0.5, 0.1],
        "epochs": [250, 750],
        "l2": [1e-4, 1e-3], 
        "all_labels":  [2**np.arange(np.log2(y.min().item()), np.log2(y.max().item()+1))]
    }

    models = []
    y_trues = []
    y_preds = []
    sample_names = []
    for ix, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(X, y, groups_CV)):
        print("Outer CV loop", ix, "...")
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_train, groups_test = groups_CV[train_index], groups_CV[test_index]

        model, _, _ = Tune(OrdinalModel, param_grid, X_train, y_train, groups_train)

        models.append(model)
        y_preds.append(model.predict(X_test))
        y_trues.append(y_test.values.reshape(-1))
        sample_names.append(groups[test_index])

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    sample_names = np.concatenate(sample_names)


    print("Acc              : %.4f" % accuracy(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("Acc(+-1)         : %.4f" % accuracy_plus_minus(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("C-ix             : %.4f" % concordance(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("C-ix(+-1)        : %.4f" % concordance_plus_minus(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print()

    random_preds = np.zeros(param_grid["all_labels"][0].shape)
    random_preds[np.isin(param_grid["all_labels"][0], np.unique(y))] = (np.unique(y, return_counts = True)[1] / len(y))
    random_preds = random_preds.reshape(1, -1).repeat(len(y), axis = 0)

    print("Random Acc       : %.4f" % accuracy(y_trues, random_preds, possible_labels = param_grid["all_labels"][0]))
    print("Random Acc(+-1)  : %.4f" % accuracy_plus_minus(y_trues, random_preds, possible_labels = param_grid["all_labels"][0]))
    print("Random C-ix      : %.4f" % 0.5)
    print("Random C-ix(+-1) : %.4f" % 0.5)

    #### WEIGHTS

    weights = np.array([m.weights_ for m in models])

    if args.save_weights_path_prefix != "":
        weight_dataframe = pd.DataFrame(
            weights, columns = X.columns,
            index = ["CV fold " +str(i) for i in range(len(weights))]
            )
        weight_dataframe.to_csv(args.save_weights_path_prefix + "weights.csv")

    if args.save_figs_path_prefix != "":
        fig, axes = weights_plot(weights, X)
        axes.set_title(y.columns[0])
        fig.tight_layout()
        fig.savefig(args.save_figs_path_prefix + "weights.svg")

        fig, axes = preds_plot(y_trues, y_preds, param_grid, sample_names)        
        fig.suptitle(y.columns[0], fontsize=32)
        fig.tight_layout()
        fig.savefig(args.save_figs_path_prefix + "predictions.svg")

    if args.save_preds_path_prefix != "":
        predictions_dataframe = pd.DataFrame(
            np.concatenate([y_trues.reshape(-1, 1), np.round(y_preds, 5)], axis = 1),
            columns = ["True label"] + ["Probability class " + str(i) for i in range(y_preds.shape[1])],
            index = sample_names
            )
        predictions_dataframe.to_csv(args.save_preds_path_prefix + "predictions.csv")





if __name__ == "__main__":
    main()
