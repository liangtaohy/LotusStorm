# -*- coding: utf-8 -*-
from framework import MLog
import os
import jieba
import jieba.analyse
import nltk
import random
import pickle

corpus_data_dir = '/Users/xlegal/Program/Work/NextLegalDev/corpus'

word_features = []


def document_features(document):
    """
    feature extractor
    :param document:
    :return:
    """
    global word_features
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def save_to_pickle(obj, filename):
    """
    save obj to pickle
    :param obj:
    :param filename:
    :return:
    """
    save_f = open(filename, "wb")
    pickle.dump(obj, save_f)


def load_documents(filename):
    """
    load corpus from file
    :param filename:
    :return: dictionary {'category':[words...]}
    """
    global word_features
    corpus_file = os.path.join(corpus_data_dir, filename)
    if not os.path.exists(corpus_file):
        #MLog.logger.fatal('corpus file not existed: ' + corpus_file)
        exit()

    file_ = open(corpus_file)
    try:
        text = file_.readline()

        documents = []

        while text:
            list_t = text.split(' ', 1)
            category = list_t[0]
            raw = list_t[1]
            #MLog.logger.debug('raw text: ' + text)
            #seg_list = jieba.cut(raw)
            seg_list = jieba.analyse.extract_tags(raw)
            documents.append((list(seg_list), category))
            text = file_.readline()

        random.shuffle(documents)
        save_to_pickle(documents, "documents.pickle")

        all_words = nltk.FreqDist(w for (words, _) in documents
                                  for w in words)

        most_common = all_words.most_common(2000)
        for (w, _) in most_common:
            word_features.append(w)

        save_to_pickle(word_features, "word_features.pickle")

        featuresets = [(document_features(d), c) for (d, c) in documents]

        #total = len(featuresets)
        train_set, test_set = featuresets[100:], featuresets[:100]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(5)

        save_to_pickle(classifier, "NaiveBayesClassifier.pickle")

        classifier = nltk.DecisionTreeClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))

        save_to_pickle(classifier, "DecisionTreeClassifier.pickle")

    finally:
        file_.close()


if __name__ == "__main__":
    file = 'corpus_findlaw_list.txt'
    load_documents(file)