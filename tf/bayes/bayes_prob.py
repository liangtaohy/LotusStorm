import json
import pickle
import os
import sys
from tf.bayes.bayes import *


def bayes_prob(cls_file, word_bag_file, input_text):
    """
    贝叶斯分类预测
    :param cls_file:
    :param word_bag_file:
    :param input_text:
    :return:
    """

    cls = pickle.load(open(cls_file, "rb"))

    word_bag = json.load(open(word_bag_file, "r", encoding="utf-8"))

    input_set = jieba_fenci(input_text, stopwords_list=stop_words_local())
    input_feature = document_features(input_set, word_bag)
    result = cls.prob_classify(input_feature)
    print("测试结果：")
    print(result.max())
    return result.max()


if __name__ == "__main__":
    cls_file = sys.argv[1]

    word_bag_file = sys.argv[2]

    input_text = sys.argv[3]

    if os.path.isfile(input_text):
        with open(input_text, 'r', encoding='utf-8') as f:
            hitted = 0
            total = 0
            for line in f:
                label = line.split(",")[0]
                sample = line.split(",")[1:]
                sample = "".join(sample)
                r = bayes_prob(cls_file, word_bag_file, input_text=sample)
                if r == int(label):
                    hitted += 1
                total += 1
                print("hit: {0}, label: {1}".format(hitted, label))
        print("hitted: {0}, total: {1}, accuracy: {2}".format(hitted, total, hitted/total))
    else:
        bayes_prob(cls_file, word_bag_file, input_text=input_text)
