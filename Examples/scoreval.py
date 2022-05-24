import numpy as np
from matplotlib import pyplot as plt

from PatternLib.probability import minDetectionCost, normalizedBayesRisk, getConfusionMatrix
from PatternLib.probability import normalizedBayesRisk
from PatternLib.validation import plotROC
from PatternLib.plotting import plot_multiple_mindcf_chart
from PatternLib.plotting import plot_multiple_mindcf_bar_chart
from PatternLib.plotting import plot_multiple_bayes_error

precision = 3


def score_eval(savetitle=None, alsoActual=False):
    labels = np.load("result/"+savetitle+"/labels.npy").astype("int32")
    scores = np.load("result/"+savetitle+"/scores.npy")
    pipe = np.load("result/"+savetitle+"/pipe.npy")
    x = np.zeros((3, scores.shape[0]))
    for i, (pip, score, label) in enumerate(zip(pipe, scores, labels)):
        s: str = pip[0]
        if s.startswith(""):
            print(pip)
            mindcf1, _ = minDetectionCost(score, label, -1, 0.5)
            mindcf2, _ = minDetectionCost(score, label, -1, 0.1)
            mindcf3, _ = minDetectionCost(score, label, -1, 0.9)
            print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")
            if alsoActual:
                priors = [0.5, 0.1, 0.9]
                ts = [-np.log(p/(1-p)) for p in priors]
                actDCFs = [normalizedBayesRisk(getConfusionMatrix(score > t, label), 1, 1, priors[i]) for i, t in enumerate(ts)]
                print(f"& {round(actDCFs[0], precision)} & {round(actDCFs[1], precision)} & {round(actDCFs[2], precision)} \\\\")
            if savetitle is not None:
                x[:, i] = [mindcf1, mindcf2, mindcf3]
    if savetitle is not None:
        np.save("tmp/" + savetitle, x)

def joint_score_eval():
    scores = np.load("result/jointResults2/jointscores.npy")
    labels = np.load("result/jointResults2/jointlabels.npy")
    for score, label in zip(scores, labels):
        mindcf1, _ = minDetectionCost(score, label, 10000, 0.5)
        mindcf2, _ = minDetectionCost(score, label, 10000, 0.1)
        mindcf3, _ = minDetectionCost(score, label, 10000, 0.9)
        print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


def plotROCscores():
    labels = np.load("result/final2/labels.npy").astype("int32")
    scores = np.load("result/final2/scores.npy")
    pipe = np.load("result/final2/pipe.npy")
    labelsSVM = np.load("result/SVMfinal2/labels.npy").astype("int32")
    scoresSVM = np.load("result/SVMfinal2/scores.npy")
    pipeSVM = np.load("result/SVMfinal2/pipe.npy")
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        if s.startswith("StandardScaler()->PCA(n_feat=7)->TiedCovarianceGMM(alpha=0.1, N=4"):
            print(pip)
            plotROC(score, label, "Tied Covariance GMM")
        if s.startswith("StandardScaler()->PCA(n_feat=7)->TiedG"):
            print(pip)
            plotROC(score, label, "Tied Gaussian")
        if s.startswith("StandardScaler()->PCA(n_feat=7)->SVM(kernel=Linear, C=0.5, regularization=5)"):
            print(pip)
            plotROC(score, label, "SVM Linear")

    for pip, score, label in zip(pipeSVM, scoresSVM, labelsSVM):
        s: str = pip[0]
        if s.startswith("StandardScaler()->PCA(n_feat=7)->SVM(kernel=Radial, C=10, regularization=1, gamma=0.05)"):
            print(pip)
            plotROC(score, label, "SVM Radial")

    # scores = np.load("result/jointResults2/jointscores.npy")
    # labels = np.load("result/jointResults2/jointlabels.npy")
    # plotROC(scores[0], labels[0], "JointModel")
    plt.plot(np.array([[0, 0], [1, 1]]))
    plt.legend()
    plt.savefig("images/ROC.eps", format='eps')
    plt.show()


def confront_bayes_error(dir="", legend=None):
    scores = np.load(f'{dir}/scores.npy')
    label = np.load(f'{dir}/labels.npy')
    pipe = np.load(f'{dir}/pipe.npy')

    plot_multiple_bayes_error(scores, label, np.linspace(-3, 3, 21), pipe, legend=legend)


if __name__ == "__main__":
    name = "logreg_mindcf"
    # score_eval(savetitle=name)
    '''    x = np.load("tmp/"+name+".npy")
    plot_multiple_mindcf_chart(x,
                               np.power(10, np.linspace(-3, 1, 51)),
                               legend=["minDCF$(\widetilde{\pi}=0.5)$", "minDCF$(\widetilde{\pi}=0.1)$", "minDCF$(\widetilde{\pi}=0.9)$"],
                               x_label="C",
                               x_scale='log',
                               savetitle=name)
    plt.show()'''
    #score_eval('20-05-2022-13-14-30')

    confront_bayes_error("./result/20-05-2022-14-32-29", legend=["SVM Poly", "SVM RBF"])
