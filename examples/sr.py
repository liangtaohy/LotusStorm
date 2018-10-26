import numpy as np
import time
X = np.ndfromtxt('images.csv', delimiter=',')
y_orig = np.ndfromtxt("labels.csv", delimiter=',', dtype=np.int8)
img_size = X.shape[1]
num_class = 10
y = np.zeros([len(y_orig), num_class])
y[np.arange(len(y_orig)), y_orig] = 1

num_train = int(len(y) * 0.8)
X_train = X[0:num_train, :]
X_test = X[num_train:-1,:]
y_train = y[0:num_train]
y_test = y[num_train:-1]


def h_elementwise(theta, X):
    phi = np.zeros([X.shape[0], theta.shape[1]])
    for i in range(X.shape[0]):
        temp = np.matmul(np.transpose(theta), X[i, :])
        temp = temp - np.amax(temp)
        temp2 = np.exp(temp)
        phi[i, :] = temp2 / np.sum(temp2)
    return(phi)


def h_vec(theta, X):
    eta = np.matmul(X, theta)
    temp = np.exp(eta - np.reshape(np.amax(eta, axis=1), [-1, 1]))
    return (temp / np.reshape(np.sum(temp, axis=1), [-1, 1]))


# not fully vectorized
def GD(theta, X_train, y_train, alpha):
    diff = h_vec(theta, X_train) - y_train
    for k in range(num_class):
        theta[:, k] -= alpha * np.squeeze(np.matmul(np.reshape(diff[:, k], [1, -1]), X_train))


def train(X_train, y_train, max_iter, alpha):
    theta = np.zeros([img_size, 10])
    for i in range(max_iter):
        GD(theta, X_train, y_train, alpha)
    return theta


max_iter = 100
alpha = 0.001
start = time.time()
theta = train(X_train, y_train, max_iter, alpha)
end = time.time()
print("time elapsed: {0} seconds".format(end - start))
pred = np.argmax(h_vec(theta, X_test), axis=1)
print("percentage correct: {0}".format(np.sum(pred == np.argmax(y_test, axis=1)) / float(len(y_test))))