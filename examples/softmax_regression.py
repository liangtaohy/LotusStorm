import numpy as np
import time

X = np.ndfromtxt('images.csv', delimiter=',')
y_orig = np.ndfromtxt("labels.csv", delimiter=',', dtype=np.int8)

# 训练样本比例
ratio = .8

num_train = int(X.shape[0] * ratio)

# 特征维度
K = X.shape[1]

# 标签数量
C = 10

# 构建one-hot向量矩阵
# 如y = 3 : [0 0 0 1 0 0 0 0 0 0]

Y = np.zeros((X.shape[0], C))

Y[range(X.shape[0]), y_orig] = 1

# 构建训练样本和测试样本
X_train = X[:num_train, :]
X_test = X[num_train:-1, :]
Y_train = Y[:num_train, :]
Y_test = Y[num_train:-1, :]


def h_elementwise(theta, x):
    """
    计算h(ɵ)：softmax假设函数
    :param theta:
    :param x: train集合
    :return:
    """
    h = np.zeros((x.shape[0], theta.shape[1]))

    for i in range(x.shape[0]):
        temp = np.matmul(np.transpose(theta), x[i, :])
        temp1 = np.exp(temp - np.amax(temp))
        h[i, :] = temp1 / np.sum(temp1)
    return h


def h_vec(theta, x):
    """
    假设函数
    :param theta: 参数矩阵
    :param x: 训练样本
    :return: h()
    """
    temp = np.matmul(x, theta)
    temp1 = np.exp(temp - np.reshape(np.amax(temp, axis=1), [-1, 1]))
    return temp1 / np.reshape(np.sum(temp1, axis=1), [-1, 1])


def GD(theta, x_train, y_train, alpha):
    """
    Gradient Descent
    :param theta: 参数矩阵
    :param x_train: 训练样本集合
    :param y_train: 标签样本集合
    :param alpha: 学习速率
    :return:
    """
    diff = h_vec(theta, x_train) - y_train
    for i in range(C):
        theta[:, i] -= alpha * np.squeeze(np.matmul(np.reshape(diff[:, i], [1, -1]), x_train))


def train(x_train, y_train, max_iter, alpha):
    theta = np.zeros([K, C])
    for i in range(max_iter):
        GD(theta, x_train, y_train, alpha)
    return theta


max_iter = 100
alpha = 0.001
start = time.time()
theta = train(X_train, Y_train, max_iter, alpha)
end = time.time()
print("time elapsed: {0} seconds".format(end - start))
pred = np.argmax(h_vec(theta, X_test), axis=1)
print("percentage correct: {0}".format(np.sum(pred == np.argmax(Y_test, axis=1)) / float(len(Y_test))))
