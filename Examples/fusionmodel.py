import os
from datetime import datetime
from typing import List
import numpy as np
from filters import filters
import PatternLib.Classifier as cl
import PatternLib.pipeline as pip
import PatternLib.preproc as prep
import PatternLib.validation as val
from modeleval import get_wine_data
from PatternLib.recalibration import Recalibration
from PatternLib.plotting import plot_multiple_bayes_error

def fusion_train(classifiers):
	train, train_labels, _, _ = get_wine_data(labels=True)
	scores = np.empty((0,), float)

	train_gauss = prep.gaussianization(train)
	K = 5
	err_rate = 0

	for classifier in classifiers:
		tr = train_gauss if isinstance(classifier, cl.GaussianClassifier) else train
		for x_tr, y_tr, x_ev, y_ev in val.kfold_split(tr, train_labels, K):
			classifier.fit(x_tr, y_tr)
			lab, ratio = classifier.predict(x_ev, True)
			err_rate += val.err_rate(lab, y_ev) / K
			scores = np.append(scores, ratio, axis=0)
	scores = scores.reshape((len(classifiers), -1))

	rec = Recalibration()
	rec_scores, rec_labels = rec.fit_transform(scores.T, train_labels)

	plot_multiple_bayes_error(rec_scores.reshape(1, train_labels.size), train_labels.reshape(1, train_labels.size), np.linspace(-3, 3, 101), np.array(["Fusion"]), legend=["Fusion"])


if __name__ == '__main__':
	classifiers = [
		cl.GaussianClassifier(),
		cl.SVM(c=1, k=1, ker='Poly', paramker=[2, 1]),
		cl.SVM(c=2, k=1, ker='Radial', paramker=[0.05]),
		cl.GaussianMixture(psi=0.01, alpha=0.1, N=3)
	]

	fusion_train(classifiers)