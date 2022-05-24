import numpy as np

import PatternLib.Classifier as cl
from PatternLib.pipeline import Pipeline, Jointer
from PatternLib.preproc import StandardScaler
from PatternLib.probability import minDetectionCost, getConfusionMatrix, normalizedBayesRisk
from PatternLib.validation import grid_search, kfold_split, err_rate
from PatternLib.recalibration import Recalibration
from modeleval import get_wine_data


def evaluatejoint():
    """
    Hyper parameter tuning and threshold calibration for the joint model
    """
    train, train_labels, test, test_labels = get_wine_data(labels=True)
    scores_f = []
    labels_f = []
    pip = Pipeline([StandardScaler(),
                    Jointer([
                        cl.GaussianClassifier(outlier_filter=True),
                        cl.SVM(c=1, k=1, ker='Poly', paramker=[2, 1]),
                        cl.SVM(c=2, k=1, ker='Radial', paramker=[0.05]),
                        cl.GaussianMixture(psi=0.01, alpha=0.1, N=10)
                    ]),
                    ])
    for hyper in grid_search({'norm_coeff': [np.power(10, x) for x in np.linspace(-3, 1, 11)], 'prior': [0.5]}):
        pip.add_step(Recalibration(**hyper))
        err_r = 0
        scores = np.empty((0,), float)
        labels = np.empty((0,), int)
        for x_tr, y_tr, x_ev, y_ev in kfold_split(train, train_labels, 5):
            pip.fit(x_tr, y_tr)
            lab, ratio = pip.predict(x_ev, True)
            err_r += err_rate(lab, y_ev)
            scores = np.append(scores, ratio, axis=0)
            labels = np.append(labels, y_ev, axis=0)
        m1, t1 = minDetectionCost(scores, labels, n_trys=-1, pi1=0.5)
        m2, t2 = minDetectionCost(scores, labels, n_trys=-1, pi1=0.1)
        m3, t3 = minDetectionCost(scores, labels, n_trys=-1, pi1=0.9)
        scores_f.append(scores)
        labels_f.append(labels)
        print(pip)
        print(t1, t2, t3)
        print(err_r / 5)
        pip.rem_step()
        print(f"{round(m1, 3)} & {round(m2, 3)} & {round(m3, 3)}")
    np.save("result/jointResults2/jointlabels", scores_f)
    np.save("result/jointResults2/jointscores", labels_f)

def test_final():
    """
    Test the chosen model and compute the DCF on the test set, this is the final evaluation for the project
    """
    train, train_labels, test, test_labels = get_wine_data(labels=True)
    thresh1 = -2.7529140657835325
    thresh2 = -2.1681219948241015
    thresh3 = -3.081358927555268
    pip = Pipeline([StandardScaler(),
                    Jointer([
                        cl.GaussianClassifier(outlier_filter=True),
                        cl.SVM(c=1, k=1, ker='Poly', paramker=[2, 1]),
                        cl.SVM(c=2, k=1, ker='Radial', paramker=[0.05]),
                        cl.GaussianMixture(alpha=0.1, psi=0.01, N=3)
                    ]),
                    StandardScaler(),
                    cl.LogisticRegression(norm_coeff=0.1)
                    ])
    pip.fit(train, train_labels)
    comp_labels = pip.predict(test, return_prob=True)[1]

    pred1 = np.where(comp_labels > thresh1, True, False)
    conf1 = getConfusionMatrix(pred1, test_labels)
    r = normalizedBayesRisk(conf1, Cfn=1, Cfp=1, pi1=.5)
    print("DCF: ", r)
    fnr = conf1[0, 1] / (conf1[0, 1] + conf1[1, 1])
    fpr = conf1[1, 0] / (conf1[1, 0] + conf1[0, 0])
    print("fnr, fpr: ", fnr, fpr)
    print("Error rate: ", err_rate(pred1, test_labels))

    pred2 = np.where(comp_labels > thresh2, True, False)
    conf2 = getConfusionMatrix(pred2, test_labels)
    r = normalizedBayesRisk(conf2, Cfn=1, Cfp=1, pi1=.1)
    print("DCF: ", r)

    pred3 = np.where(comp_labels > thresh3, True, False)
    conf3 = getConfusionMatrix(pred3, test_labels)
    r = normalizedBayesRisk(conf3, Cfn=1, Cfp=1, pi1=.9)
    print("DCF: ", r)


if __name__ == "__main__":
    evaluatejoint()
