import numpy as np
import math
import time
X = np.ndfromtxt('images.csv', delimiter=',')
y = np.ndfromtxt("labels.csv", delimiter=',', dtype=np.int8)
img_size = X.shape[1]
print(img_size)
print(X.shape)
print(y.shape)

ind = np.logical_or(y == 1, y == 0)
print(ind)
X = X[ind, :]
y = y[ind]

num_train = int(len(y) * 0.8)
X_train = X[0:num_train, :]
X_test = X[num_train:-1,:]
y_train = y[0:num_train]
y_test = y[num_train:-1]

print("num_train: %d" % num_train)
print(X_train)


def h1(theta, x):
    sum = 0.0
    for i in range(len(x)):
        sum -= theta[i] * x[i]
    return 1 / (1 + math.exp(sum))


def h2(theta, x):
    return 1 / (1 + np.exp(np.dot(theta, x)))


def h(theta, x):
    return 1 / (1 + np.exp(-np.dot(theta, x)))


def GD_elementwise(theta, X_train, y_train, alpha):
    diff_arr = np.zeros([len(y_train)])
    for m in range(len(y_train)):
        diff_arr[m] = h(theta, X_train[m, :]) - y_train[m]
    for j in range(len(theta)):  # feature j
        s = 0.0
        for m in range(len(y_train)):  # sample m
            s += diff_arr[m] * X_train[m, j]
        theta[j] = theta[j] - alpha * s


def train_elementwise(X_train, y_train, max_iter, alpha):
    theta = np.zeros([img_size])
    for i in range(max_iter):
        GD_elementwise(theta, X_train, y_train, alpha)
    return theta


def h_vec(theta, X):
    return 1 / (1 + np.exp(-np.matmul(X, theta)))


def GD_better(theta, X_train, y_train, alpha):
    diff_arr = h_vec(theta, X_train) - y_train
    for j in range(len(theta)):
        theta[j] = theta[j] - alpha * np.dot(diff_arr, X_train[:, j])


def train_better(X_train, y_train, max_iter, alpha):
    theta = np.zeros([img_size])
    for i in range(max_iter):
        GD_better(theta, X_train, y_train, alpha)
    return theta


max_iter = 10
alpha = 0.01
start = time.time()
theta = train_better(X_train, y_train, max_iter, alpha)
end = time.time()
print("time elapsed: {0} seconds".format(end - start))
pred = (np.sign(h_vec(theta, X_test) - 0.5) + 1) / 2
print("percentage correct: {0}".format(np.sum(pred == y_test) / len(y_test)))

exit(0)

max_iter = 10

alpha = 0.01

start = time.time()

theta = train_elementwise(X_train, y_train, max_iter, alpha)
end = time.time()

print("theta size: %d " % len(theta))

print(X_test)

print(y_test)
print("time elapsed: {0} seconds".format(end - start))
pred = (np.sign(h_vec(theta, X_test) - 0.5) + 1) / 2
print(pred)
print("percentage correct: {0}".format(np.sum(pred == y_test) / len(y_test)))


