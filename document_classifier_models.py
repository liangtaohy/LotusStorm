# -*- coding: utf-8 -*-
from framework import MLog
import os
import jieba
import jieba.analyse
import nltk
import random
import pickle

corpus_data_dir = './sample'

word_features = []

categories_arr = []

feature_sets_chi = []


def document_features(document):
    """
    feature extractor
    :param document:
    :return:
    """
    global word_features
    document_words = set(document)
    features = {}
    for word, freq in word_features:
        if word in document_words:
            features[word] = freq
        else:
            features[word] = 0
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


def feature_selection_chi(documents, all_words):
    word_segs_A = nltk.FreqDist()
    word_segs_B = nltk.FreqDist()
    word_degree_C = nltk.FreqDist()
    word_degree_D = nltk.FreqDist()
    category = []
    all_words = set(all_words)
    N = len(all_words)
    for (words, c) in documents:
        set_words = set(words)
        category.append(c)
        for w in all_words:
            if w in set_words:
                word_segs_A[w + 'is' + c] += 1
                word_segs_B[w] += 1
            else:
                word_degree_C['not' + w + c] += 1
                word_degree_D[w] += 1
    features = nltk.FreqDist()
    for c in category:
        ws = nltk.FreqDist()
        for w in all_words:
            A = word_segs_A[w + 'is' + c]
            B = word_segs_B[w] - A
            C = word_degree_C['not' + w + c]
            D = word_degree_D[w] - C
            E = N*((A*D - B*C) * (A*D - B*C)) / ((A+C) * (A + B) * (B+D) * (C + D))
            ws[w] = E
        for (w, _) in ws.most_common(1000):
            features[w] += 1
    return features


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

        jieba.analyse.set_stop_words('./framework/stop_words_for_tags.txt')
        while text:
            list_t = text.split(' ', 1)
            category = list_t[0]
            raw = list_t[1]

            # feature extractor (特征词抽取。对于短文本而言，这个就够用了)
            seg_list = jieba.analyse.extract_tags(raw, topK=30, withWeight=False)
            documents.append((list(seg_list), category))
            text = file_.readline()

        random.shuffle(documents)

        save_to_pickle(documents, "documents.pickle")

        # bag of words 词袋
        all_words = nltk.FreqDist(w for (words, _) in documents
                                  for w in words)

        # all_words = feature_selection_chi(documents, all_words)

        # feature selection 特征选择（TF-词频）
        N = all_words.N()
        most_common = all_words.most_common(2000)
        for (w, freq) in most_common:
            word_features.append((w, freq/N))

        print(word_features)

        save_to_pickle(word_features, "word_features.pickle")

        featuresets = [(document_features(d), c) for (d, c) in documents]

        train_set, test_set = featuresets[500:], featuresets[:500]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(nltk.classify.accuracy(classifier, test_set))
        classifier.show_most_informative_features(5)

        """
        Test Case
        """
        labels = classifier.labels()
        for label in labels:
            p = classifier.prob_classify(document_features(jieba.analyse.extract_tags("全国人民代表大会常务委员会关于特赦确实改恶从善的罪犯的决定［失效］"))).prob(label)
            print("全国人民代表大会常务委员会关于特赦确实改恶从善的罪犯的决定［失效］:%s-%f" % (label, p))

        test_text = "最高人民法院关于审理期货纠纷案件若干问题的规定"
        test_feature = document_features(jieba.analyse.extract_tags(test_text))
        if len(test_feature):
            for label in labels:
                p = classifier.prob_classify(test_feature).prob(label)
                print("%s:%s-%f" % (test_text, label, p))
        print(classifier.prob_classify(test_feature).max())
        save_to_pickle(classifier, "NaiveBayesClassifier.pickle")

        """
        Too Slow!!!
        """
        if False:
            classifier = nltk.DecisionTreeClassifier.train(train_set)
            print(nltk.classify.accuracy(classifier, test_set))

        save_to_pickle(classifier, "DecisionTreeClassifier.pickle")

    finally:
        file_.close()


if __name__ == "__main__":
    file = 'corpus_findlaw_list.txt'
    load_documents(file)