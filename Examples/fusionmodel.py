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


def fusion_train(classifiers, rec, std_scaler):
	train, train_labels, _, _ = get_wine_data(labels=True)
	scores = np.empty((0,), float)
	labels = np.empty((0,), float)

	train_gauss = prep.gaussianization(train)
	K = 5
	err_rate = 0

	train_scaled = std_scaler.fit_transform(train, train_labels)

	i = 0
	for classifier in classifiers:
		tr = train_gauss if isinstance(classifier, cl.GaussianClassifier) else train_scaled
		for x_tr, y_tr, x_ev, y_ev in val.kfold_split(tr, train_labels, K):
			classifier.fit(x_tr, y_tr)
			x_ev = std_scaler.transform(x_ev)

			lab, ratio = classifier.predict(x_ev, True)
			err_rate += val.err_rate(lab, y_ev) / K
			scores = np.append(scores, ratio, axis=0)
			if i == 0:
				labels = np.append(labels, y_ev)
		i += 1
	scores = scores.reshape((len(classifiers), -1))

	rec_scores = rec.fit_transform(scores.T, labels)
	plot_multiple_bayes_error(rec_scores.reshape(1, train_labels.size), labels.reshape(1, labels.size), np.linspace(-4, 4, 101), np.array(["Fusion"]), legend=["Fusion"])


def fusion_test(classifiers, rec, std_scaler):
	train, _, test, test_labels = get_wine_data(labels=True)

	test_scaled = std_scaler.transform(test)
	test_gauss = prep.gaussianization(np.append(train, test, axis=0))[train.shape[0]:, :]

	scores = np.empty((0,), float)

	for classifier in classifiers:
		test_ = test_gauss if isinstance(classifier, cl.GaussianClassifier) else test_scaled
		lab, ratio = classifier.predict(test_, True)
		print(classifier, f"error rate: {val.err_rate(lab, test_labels)}")
		scores = np.append(scores, ratio, axis=0)
	scores = scores.reshape((len(classifiers), -1))

	rec_scores = rec.transform(scores.T)

	print("Recalibration:")
	print(f"Fusion error rate: {val.err_rate(rec_scores > 0, test_labels)}")

	plot_multiple_bayes_error(rec_scores.reshape(1, rec_scores.size), test_labels.reshape(1, test_labels.size), np.linspace(-4, 4, 101), np.array(["Fusion"]), legend=["Fusion"])


if __name__ == '__main__':
	# State of the fusion (classifier, recalibration, std_scale, training_data_for_gauss)
	classifiers = [
		cl.SVM(c=1, k=1, ker='Poly', paramker=[2, 1]),
		cl.SVM(c=2, k=1, ker='Radial', paramker=[0.5]),
		cl.GaussianMixture(psi=0.01, alpha=0.1, N=3),
		cl.GaussianClassifier()
	]

	classifiers = [
		cl.GaussianClassifier(),
		cl.GaussianClassifier(),
		cl.GaussianClassifier(),
		cl.GaussianClassifier()
	]

	rec = Recalibration()

	std_scale = prep.StandardScaler()

	# Train
	fusion_train(classifiers, rec, std_scale)

	#Test
	fusion_test(classifiers, rec, std_scale)
