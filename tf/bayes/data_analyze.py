# -*- coding: utf-8 -*-
import numpy as np
import json
import pandas
import matplotlib.pyplot as plt


def get_unrelative_words(train_set_file, word_bag_file):
    """
    获取疑似不相关词
    :param train_set_file 训练样本集合，json数据，已分词，结构为[({tokens}, label)]
    :param word_bag_file 词袋集合,json数据，结构为[tokens]
    :return: unrelative_words 返回与分类不相关的词
    """
    train_set = json.load(open(train_set_file, "r", encoding="utf-8"))
    word_bag = json.load(open(word_bag_file, "r", encoding="utf-8"))

    num_train = len(train_set)
    num_word = len(word_bag)

    X = np.zeros((num_train, num_word), dtype=np.int)

    labels = []

    for i in range(num_train):
        labels.append(train_set[i][1])
        for j in range(num_word):
            if word_bag[j] in train_set[i][0]:
                X[i][j] += 1

    hist = pandas.value_counts(labels)
    hist = pandas.Series(hist)
    hist.plot(kind='bar')
    plt.show()

    print(pandas.value_counts(labels))

    labels = set(labels)

    print("{0} samples per label".format(num_train / len(labels)))

    print(X)

    Z = np.zeros((X.shape[1],))

    print(X.shape)

    df = np.sum(X, axis=0)

    sorted_df_indices = np.argsort(df)

    print(sorted_df_indices)

    average_per_label = num_train / len(labels)

    up = int(0.6 * num_train)
    down = int(average_per_label / 3)
    print("up:{0},down:{1}".format(up, down))
    clean = [word_bag[i] for i in sorted_df_indices if df[i] > down and df[i] < up]
    unrelative_words = [word_bag[i] for i in sorted_df_indices if df[i] <= down or df[i] >= up]

    print([word_bag[i] for i in sorted_df_indices if df[i] <= down])
    return unrelative_words


if __name__ == "__main__":
    unrelative_words = get_unrelative_words("./.train_set.json", "./.word_bag.json")
    json.dump(unrelative_words, open('unrelative_words.json', "w", encoding="utf-8"), ensure_ascii=False)