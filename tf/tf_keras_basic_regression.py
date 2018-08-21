import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

boston_housing = keras.datasets.boston_housing

(train_data, train_label), (test_data, test_labels) = boston_housing.load_data()

print("Training set: {}".format(train_data.shape))  # 404 个样本, 13 个特征
print("Testing set:  {}".format(test_data.shape))   # 102 个样本, 13 个特征
print("Training label: {}".format(train_label.shape))
print("Testing label:  {}".format(test_labels.shape))

# Shuffle the training set
order = np.argsort(np.random.random(train_label.shape))
train_data = train_data[order]
train_label = train_label[order]

print(train_data[0])  # 查看下原始数据

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

print(train_label[0:10])  # 显示前10个的标签，即价格

# 数据正则化 （使用均值和标准差）
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(mean)
print(train_data[0])  # 显示第一条数据样本的正则化结果

# 构建模型


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = create_model()
model.summary()


# 训练模型

# 显示训练进度（每完成一个epoch，则打印一个点）
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 500

# 存储训练的统计数据

history = model.fit(train_data, train_label, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()


plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

# 预测
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

print(test_predictions)
