import json
import pickle
import nltk
import sys
from bayes import *


def bayes_prob(cls_file, word_bag_file, input_text):


    cls = pickle.load(open(cls_file, "rb"))

    word_bag = json.load(open(word_bag_file, "r", encoding="utf-8"))

    input_set = jieba_fenci(input_text, stopwords_list=stop_words_local())
    input_feature = document_features(input_set, word_bag)
    result = cls.prob_classify(input_feature)
    print("测试结果：")
    print(result.max())


if __name__ == "__main__":
    cls_file = sys.argv[1]

    word_bag_file = sys.argv[2]

    bayes_prob(cls_file, word_bag_file, input_text="如果创始人和公司未签署本条款清单并交付给投资人，其将于2017年12月18日18:30分失效")