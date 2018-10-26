import numpy as np

sigma = 1
mu = 0
np.array((2))

# 构造样本数据 X, Y
X = np.random.normal(loc=mu, scale=sigma, size=(1000, 2))
W = np.array([[2, -3.4]])
W = np.transpose(W)
b = 4.2

Y = np.matmul(X, W) + b

# 加上随机噪声
Y = Y + np.random.normal(0, 0.1, size=Y.shape)

# debug message
print(Y)
print(W.shape)
print(W)
print(X.shape)
print(Y.shape)

print(abs(np.mean(X)))
print(np.std(X, ddof=1))


def show_pic(X, sigma):
    """
    show picture

    :param X:
    :param sigma:
    :return:
    """
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(211)
    count, bins, ignored = plt.hist(X, 30, normed=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins-mu)**2 / (2 * sigma**2)), linewidth=2, color='r')

    plt.subplot(212)
    plt.scatter(X[:, 1], Y)

    plt.show()


# 使用正态分布初始化权重矩阵
W = np.random.normal(0, 0.01, size=W.shape)

# 初始化bias
b = np.random.normal(size=(1,))


def mini_batch(X, Y, batch_size):
    """
    mini batch iter

    :param X: input matrix
    :param Y: input label matrix
    :param batch_size: batch size
    :return: mini batch x, mini batch y
    """
    num_examples = len(X)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        indexes = indices[i:min(i + batch_size, num_examples)]
        yield X[indexes,:], Y[indexes]


def linear(x, w, b):
    """
    线性计算: X*W + b

    :param x: input matrix
    :param w: weight matrix
    :param b: bias
    :return:
    """
    return np.matmul(x, w) + b


def square_loss(y_hat, y):
    """
    均方误差

    :param y_hat:
    :param y:
    :return:
    """
    return (y_hat - y) ** 2 / 2


def sgd(w, b, lr, batch_x, batch_y, y_hat):
    """
    随机梯度下降算法

    :param w: weights
    :param b: bias
    :param lr: learning rate
    :param batch_x: mini batch x
    :param batch_y: mini batch y
    :param y_hat: predicted y
    :return: weights, b
    """
    w = w - lr * np.matmul(np.transpose(batch_x), y_hat - batch_y) / batch_y.shape[0]
    b = b - lr * np.sum((y_hat - batch_y)) / batch_y.shape[0]
    return w, b


# 开始训练
num_epochs = 50
batch_size = 10
lr = 0.03
loss = square_loss
net = linear

for i in range(num_epochs):
    for batch_x, batch_y in mini_batch(X, Y, batch_size):
        y_hat = linear(batch_x, W, b)
        W, b = sgd(W, b, lr, batch_x, batch_y, y_hat)
    train = loss(linear(X, W, b), Y)
    print('epoch %d, loss %f' % (i + 1, np.mean(train)))

# 输出weight, bias
print("weight, bias")
print(W, b)
