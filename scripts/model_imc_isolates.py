import torch
torch.set_num_threads(4)
import argparse
from utils import *
import pandas as pd
import numpy as np
    
def read_data(path):
    dataframe = pd.read_csv(path,index_col=0)
    groups = dataframe.index
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, [-1]]
    return X, y, groups


def main():
    class CustomFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        description="Script for training ordinal regression models on preprocessed MALDI-TOF mass spectra (LOOCV)",
        formatter_class=CustomFormatter,
    )

    parser.add_argument("data_path", type=str, metavar="data_path", help="path to data file")
    parser.add_argument("isolates_data_path", type=str, metavar="isolates_data_path", help="path to isolates data file")
    parser.add_argument(
        "--save_preds_path_prefix",
        type=str,
        default="",
        help="Prefix path for the file to save preds.csv to. Default \'\', which means don't save"
    )


    args = parser.parse_args()

    X, y, groups = read_data(args.data_path)

    X_isolates, y_isolates, groups_isolates = read_data(args.isolates_data_path)


    column_select_on_overlap = np.array([
        np.isclose(orig_X_time, X_isolates.columns.values.astype(float), atol=1).any()
        for orig_X_time in X.columns.values.astype(float)
    ])
    X = X.iloc[:, column_select_on_overlap]

    groups_CV = split_groups_CV(
        groups,
        ["Lineage", "Strain"],
        ["Lineage", "Treatment", "Strain"],
        print_ = True
    )

    param_grid = {
        "lr": [1, 0.5, 0.1],
        "epochs": [250, 750],
        "l2": [1e-3, 1e-4],
        "all_labels":  [2**np.arange(np.log2(y.min().item()), np.log2(y.max().item()+1))]
    }

    y_trues = []
    y_preds = []
    sample_names = []

    model, _, _ = Tune(OrdinalModel, param_grid, X, y, groups)

    all_labels = 2**np.arange(np.log2(y.min().item()), np.log2(y.max().item()+1))
    throwaway = np.isin(y_isolates, all_labels).reshape(-1)
    y_isolates = y_isolates.iloc[throwaway]
    X_isolates = X_isolates.iloc[throwaway]
    groups_isolates = groups_isolates[throwaway]

    y_trues = []
    y_preds = []
    sample_names = []
    for i in np.unique(groups_isolates):
        pred = model.predict(X_isolates.iloc[groups_isolates == i]).mean(0)
        true = y_isolates.values.reshape(-1)[groups_isolates == i][0]

        y_preds.append(pred)
        y_trues.append(true)
        sample_names.append(i)

    y_trues = np.stack(y_trues)
    y_preds = np.array(y_preds)
    sample_names = np.array(sample_names)


    print("Acc              : %.4f" % accuracy(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("Acc(+-1)         : %.4f" % accuracy_plus_minus(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("C-ix             : %.4f" % concordance(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print("C-ix(+-1)        : %.4f" % concordance_plus_minus(y_trues, y_preds, possible_labels = param_grid["all_labels"][0]))
    print()

    random_preds = np.zeros(param_grid["all_labels"][0].shape)
    ys = y.values.reshape(-1)
    random_preds[np.isin(param_grid["all_labels"][0], np.unique(ys))] = (np.unique(ys, return_counts = True)[1] / len(ys))
    random_preds = random_preds.reshape(1, -1).repeat(len(y_trues), axis = 0)

    print("Random Acc       : %.4f" % accuracy(y_trues, random_preds, possible_labels = param_grid["all_labels"][0]))
    print("Random Acc(+-1)  : %.4f" % accuracy_plus_minus(y_trues, random_preds, possible_labels = param_grid["all_labels"][0]))
    print("Random C-ix      : %.4f" % 0.5)
    print("Random C-ix(+-1) : %.4f" % 0.5)

    if args.save_preds_path_prefix != "":
        predictions_dataframe = pd.DataFrame(
            np.concatenate([y_trues.reshape(-1, 1), np.round(y_preds, 5)], axis = 1),
            columns = ["True label"] + ["Probability class " + str(i) for i in range(y_preds.shape[1])],
            index = sample_names
            )
        predictions_dataframe.to_csv(args.save_preds_path_prefix + "predictions.csv")



if __name__ == "__main__":
    main()
