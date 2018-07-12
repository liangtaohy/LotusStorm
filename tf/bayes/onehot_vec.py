# -*- coding: utf-8 -*-
# 构造文档向量集合feature和标签集合Y
# feature由one-hot向量表示。feature的行数为训练样本的数量。每一行对应样本的one-hot向量
# Y则是一维数组，其元素个数与feature的行数相同。Y[i]即为第i个样本的类别标签
# :author Liang Tao <liangtaohy@163.com>
#

import numpy as np
from bayes import stop_words_local, jieba_fenci
import json
import pickle


stop_words = stop_words_local()


def to_onhot_vec(document, word_bag):
    words = jieba_fenci(document, stop_words)

    words = set(words)

    feature = np.zeros((len(word_bag,)))

    for i in range(len(word_bag)):
        if word_bag[i] in words:
            feature[i] = 1

    return feature


def onhot_vecs(train_set_json_file, word_bag_json_file):
    train_set = json.load(open(train_set_json_file, "r", encoding="utf-8"))
    word_bag = json.load(open(word_bag_json_file, "r", encoding="utf-8"))

    X = [sample[0] for sample in train_set]
    Y = [sample[1] for sample in train_set]


    feature = np.zeros((len(X), len(word_bag)))

    for i in range(len(X)):
        for j in range(len(word_bag)):
            if word_bag[j] in X[i]:
                feature[i][j] = 1

    return feature, np.array(X), np.array(Y)


if __name__ == "__main__":
    feature, X, Y = onhot_vecs("./.train_set.json", "./.word_bag.json")
    pickle.dump(feature, open("onehot_doc_feature.pickle", "wb"))
    pickle.dump(Y, open("onehot_doc_Y.pickle", "wb"))

    print(feature.shape)
    print(feature)
    print(X)
    print(Y)