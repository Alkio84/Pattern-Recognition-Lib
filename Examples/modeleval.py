import os
from datetime import datetime
from typing import List
import numpy as np
from filters import filters
import PatternLib.Classifier as cl
import PatternLib.pipeline as pip
import PatternLib.preproc as prep
import PatternLib.validation as val

from PatternLib.checco_gmm import GMM_checco

save_logs = True
all_scores = []
all_pipes = []
all_labels = []

attributes = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


def get_wine_data(path_train="./../Data/Train.txt", path_test="./../Data/Test.txt", labels=False):
    """
    Get all pulsar data and divide into labels
    """
    train_data = np.loadtxt(path_train, delimiter=",")
    test_data = np.loadtxt(path_test, delimiter=",")
    train_data, train_labels, test_data, test_labels = train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    if labels:
        return train_data, train_labels, test_data, test_labels
    else:
        return train_data, test_data


def kfold_test(gauss=False):
    """
    Test all models with kfold strategy
    """
    train, train_labels, test, test_labels = get_wine_data(labels=True)
    if gauss:
        train = prep.gaussianization(train)
    pipe_list: List[pip.Pipeline]
    preprocessing_pipe_list = [
        pip.Pipeline([prep.StandardScaler()]),
        # pip.Pipeline([prep.Pca(10)]),
        # pip.Pipeline([prep.Pca(9)]),
        # pip.Pipeline([prep.Pca(8)]),
    ]

    # Gaussian
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.GaussianClassifier, None, train, train_labels)
    # Naive Bayes
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.NaiveBayes, None, train, train_labels)
    # Tied Gaussian
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.TiedGaussian, None, train, train_labels)
    #   Logistic regression
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'norm_coeff': [0.1, 0.5, 1]}):
            test_model(pipe, cl.LogisticRegression, hyper, train, train_labels)
    #   SVM no kern
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [0.1, 1, 5, 10], 'k': [1, 5], 'ker': [None]}):
            test_model(pipe, cl.SVM, hyper, train, train_labels)
    #   SVM Poly
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [0.5, 1], 'k': [0, 1], 'ker': ['Poly'], 'paramker': [[2, 1]]}):
            test_model(pipe, cl.SVM, hyper, train, train_labels)
    #   SVM Radial
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'paramker': [[0.5]], 'c': [np.power(10, x) for x in np.linspace(-1, 2, 51)], 'k': [1], 'ker': ['Radial'] }):
            test_model(pipe, cl.SVM, hyper, train, train_labels, True)
    #   Gaussian Mixture tied
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'tied': [True], 'alpha': [0.1], 'N': [0, 1, 2]}):
            test_model(pipe, cl.GaussianMixture, hyper, train, train_labels)
    #   Gaussian Mixture diag
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'diag': [False], 'alpha': [0.1], 'N': [0, 1, 2]}):
            test_model(pipe, cl.GaussianMixture, hyper, train, train_labels)
    #   Checco Mixture psi
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'psi': [0.01], 'alpha': [0.1], 'n_components': [8, 16, 32]}):
            test_model(pipe, GMM_checco, hyper, train, train_labels)

def test_model(pipe: pip.Pipeline, mod, hyper: dict, train: np.ndarray, train_labels: np.ndarray, outliers_filter=False):
    """
    KFold wrapper
    """
    model = mod(**hyper if hyper else {})
    pipe.add_step(model)
    err_rate = k_test(pipe, train, train_labels, 5, outliers_filter)
    print(f"{pipe}:{err_rate}")
    pipe.rem_step()


def k_test(pipe, train, train_labels, K, outliers_filter=False):
    """
    Kfold on model and save scores and labels into files
    """
    err_rate = 0
    scores = np.empty((0,), float)
    labels = np.empty((0,), int)
    for x_tr, y_tr, x_ev, y_ev in val.kfold_split(train, train_labels, K):
        mask = np.ones(x_tr.shape[0], dtype=bool)
        if outliers_filter:
            x_tr_tmp = prep.StandardScaler().fit_transform(x_tr, y_tr)
            mask = prep.filter_outliers(x_tr_tmp, y_tr, filters)
        pipe.fit(x_tr[mask, :], y_tr[mask])
        lab, ratio = pipe.predict(x_ev, True)
        err_rate += val.err_rate(lab, y_ev) / K
        scores = np.append(scores, ratio, axis=0)
        labels = np.append(labels, y_ev, axis=0)
    save_scores(scores, pipe, labels)
    return err_rate


def save_scores(score, pipe, label):
    all_scores.append(score)
    all_pipes.append(pipe.__str__())
    all_labels.append(label)
    np.save(f"{logfile}/scores", np.vstack(all_scores))
    np.save(f"{logfile}/labels", np.vstack(all_labels))
    np.save(f"{logfile}/pipe", np.vstack(all_pipes))


logfile = None

if save_logs:
    logfile = datetime.now().strftime("result/%d-%m-%Y-%H-%M-%S")

if __name__ == "__main__":
    if save_logs:
        os.mkdir(logfile)
    kfold_test()



