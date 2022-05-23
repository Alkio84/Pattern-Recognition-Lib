import numpy as np
from PatternLib.Classifier import LogisticRegression
from PatternLib.pipeline import Pipe


class Recalibration(Pipe):
    def __init__(self, prior=0.5, norm_coeff=10**-5):
        self.log_reg = LogisticRegression(norm_coeff=norm_coeff, prior=prior)

    def fit(self, x, y):
        x = x.reshape(x.size, 1)
        self.log_reg.fit(x, y)

    def transform(self, x):
        x = x.reshape(x.size, 1)

        w, b_, prior = self.log_reg.w, self.log_reg.b, self.log_reg.prior
        b = b_ - (np.log(prior/(1 - prior)))
        return np.dot(w, x.T) + b

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def __str__(self):
        return f'Recalibration (prior: {self.log_reg.prior}, norm_coeff: {self.log_reg.norm_coeff})'

