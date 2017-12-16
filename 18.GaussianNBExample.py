#18.GaussianNBExample.py

import sklearn.naive_bayes

import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = sklearn.naive_bayes.GaussianNB()
clf.fit(X, Y)
sklearn.naive_bayes.GaussianNB(priors=None)
print(clf.predict([[-0.8, -1]]))

clf_pf = sklearn.naive_bayes.GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
sklearn.naive_bayes.GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))
