import numpy as np
from PatternLib.recalibration import Recalibration
from PatternLib.plotting import plot_multiple_bayes_error

if __name__ == "__main__":
    rec = Recalibration(prior=0.1)

    scores_gmm = np.load(f'./result/gmm_8_16_32_checco/scores.npy')[0]
    labels_gmm = np.load(f'./result/gmm_8_16_32_checco/labels.npy')[0]
    pipe_gmm = np.load(f'./result/gmm_8_16_32_checco/pipe.npy')[0]

    scores_mvg = np.load(f'./result/mvg/scores.npy')[0]
    labels_mvg = np.load(f'./result/mvg/labels.npy')[0]
    pipe_mvg = np.load(f'./result/mvg/pipe.npy')[0]

    rec_scores_gmm = rec.fit_transform(scores_gmm, labels_gmm)
    rec_scores_mvg = rec.fit_transform(scores_mvg, labels_mvg)

    plot_multiple_bayes_error(np.array([rec_scores_mvg, rec_scores_gmm]), np.array([labels_mvg, labels_gmm]), np.linspace(-4, 4, 101), np.array([pipe_mvg, pipe_gmm]), legend=["MVG R", "GMM-8 R"])

    '''    rec = Recalibration(prior=0.5)

    scores = np.load(f'./result/poly_rbf/scores.npy')
    labels = np.load(f'./result/poly_rbf/labels.npy')
    pipes = np.load(f'./result/poly_rbf/pipe.npy')

    rec_scores_1 = rec.fit_transform(scores[0], labels[0]).T
    rec_scores_2 = rec.fit_transform(scores[1], labels[1]).T

    plot_multiple_bayes_error(np.array([rec_scores_1, rec_scores_2]), labels, np.linspace(-4, 4, 101), pipes, legend=["SVM Poly R", "SVM RBF R"])'''


