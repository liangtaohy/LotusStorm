import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt

samples = np.loadtxt('/Users/xlegal/Desktop/heyihan_guquan.csv', delimiter=',', unpack=True)


X = samples[0]
Y = samples[1]

order = np.argsort(np.random.random(X.shape))

X = X[order]
Y = Y[order]

mean = X.mean(axis=0)
std = X.std(axis=0)

X = (X - mean) / std
#Y = (Y - mean) / std

print(X.shape)
print(Y.shape)

plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:180], Y[:180]
X_test, Y_test = X[180:], Y[180:]

model = keras.Sequential()

model.add(keras.layers.Dense(input_dim=1, units=1))

model.compile(loss='mse', optimizer='sgd')

for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost: ", cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

print(np.mean(Y, axis=0))
print(np.std(Y, axis=0))
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
