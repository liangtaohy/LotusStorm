# -*- coding: utf-8 -*-


import numpy as np
import pickle
import time


X = pickle.load(open("../bayes/onehot_doc_feature.pickle", "rb"))

Y = pickle.load(open("../bayes/onehot_doc_Y.pickle", "rb"))

test_X = X[550:605]
test_Y = Y[550:605]

X = X[:550]
Y = Y[:550]

num_test = len(test_X)
num_train = len(X)

dist = np.zeros((num_test, num_train))

for i in range(num_test):
    for j in range(num_train):
        dist[i][j] = np.sqrt(np.sum(np.square(test_X[i, :] - X[j, :])))

indices = np.argsort(dist, axis=1)

K = 5

closest_k = Y[indices][:, :K].astype(int)

print(closest_k)

Y_pred = np.zeros_like(test_Y)

for i in range(num_test):
      Y_pred[i] = np.argmax(np.bincount(closest_k[i,:]))


accuracy = (np.where(test_Y - Y_pred == 0)[0].size) / float(num_test)
print('Prediction accuracy: {}%'.format(accuracy*100))
