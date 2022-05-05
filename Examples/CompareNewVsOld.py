import PatternLib.Classifier as c
import PatternLib.validation as v

import numpy as np
import sklearn.datasets


# Usami solo come test per questi 2 classificatori


def split_db_2to1(D, L, seed=0):
    # 100 samples for training and 50 samples for evaluation
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    # DTR and LTR training data and labels
    # DTE and LTE evaluation data and labels
    DTR = D[:, idxTrain].T
    DTE = D[:, idxTest].T
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def load_iris_binary_split():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

    D = D[:, L != 0]
    L = L[L != 0]
    L[L == 2] = 0

    return split_db_2to1(D, L)


def compare_svm():
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            for prior in [None]:
                svm = c.SVM(k=K, c=C, prior=prior)
                svm.fit(DTR, LTR)
                pred, _ = svm.predict(DTE)

                print(f"(K: {K} C:{C} prior:{prior}) err:{v.err_rate(pred, LTE) * 100}")
            print("==============================")


def compare_log_reg():
    (DTR, LTR), (DTE, LTE) = load_iris_binary_split()

    for norm_coeff in [0, 10**-6, 10**-3, 1.0]:
        for prior in [None, 0.1, 0.5, 0.9]:
            lg = c.LogisticRegression(norm_coeff=norm_coeff, prior=prior)
            lg.fit(DTR, LTR)
            pred, _ = lg.predict(DTE)

            print(f"(norm_coeff: {norm_coeff} prior:{prior}) err:{v.err_rate(pred, LTE) * 100}")
        print("==============================")


if __name__ == '__main__':
    #compare_log_reg()
    compare_svm()


