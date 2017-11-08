"""
许可证聚类分析
基于sklearn库
Target:
生成可视化的聚类图
支持Kmeans聚类算法
"""
import os
import pymysql
import math
import time
import re
from wordcloud import WordCloud
import PIL
import numpy
import nltk, re, pprint
import matplotlib.pyplot as plt
from law import JiebaTokenizer
from law import DocType
from law import GovDict
from law import Settings as settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer

class LicenseCluster:
    def __init__(self):
        self.v = '0.0.1'
        self.tokenizer = JiebaTokenizer.JiebaTokenizer(os.path.dirname(os.path.realpath(__file__)) +
                                                       '/../framework/stop_words_jieba.utf8.txt')
        self.stopList = ['经营', '生产', '服务', '业务', '临时']

    def segments_from_file(self, filepath):
        fp = open(filepath)
        seg_file = filepath + '.seg'
        output_file = open(seg_file, 'w')
        segs = []
        while 1:
            line = fp.readline()
            if not line:
                break
            tokens = self.tokenizer.tokenize(line)
            words = []
            for (word, flag) in tokens:
                w = word.strip()
                if len(w):
                    if w.find('许可证') == -1 and w != '企业' and w != '中华人民共和国' and w != '业务':
                        words.append(word)
                        segs.append(word)
            if len(words):
                output_file.write(" ".join(words) + "\n")
        fp.close()
        output_file.close()

        #fdist = nltk.FreqDist(segs)
        #fdist.plot(100, cumulative=False)

        return seg_file

    def build_word_cloud(self, filepath):
        seg_file = self.segments_from_file(filepath)
        text = open(seg_file).read()
        wordcloud = WordCloud(font_path='/System/Library/Fonts/PingFang.ttc').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def tfidf(self, dataset, n_features=1000):
        """
        trans dataset into tfidf matrix
        :param dataset:
        :param n_features:
        :return:
        """
        vec = TfidfVectorizer(max_features=n_features, use_idf=True)
        X = vec.fit_transform(dataset)
        print(X)
        return X, vec

    def train(self, X, vec, true_k=10, minibatch=False, showLable=False):
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1, verbose=False)
        km.fit(X)
        if showLable:
            print("Top terms per cluster:")
            print(vec.get_stop_words())
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            terms = vec.get_feature_names()
            for i in range(true_k):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :1]:
                    print(' %s' % terms[ind], end='')
                print()
        return -km.score(X)

    def train_kmeans(self, X, vec, true_k=10, minibatch=False, showLable=False):
        km = KMeans(n_clusters=true_k, init='k-means++', n_init=1, max_iter=1000, verbose=False)
        km.fit(X)
        print("Top terms per cluster:")
        print(vec.get_stop_words())
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vec.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :3]:
                print(' %s' % terms[ind], end='')
            print()

    def loadset(self, filepath):
        """
        load data into set
        :return:
        """
        fp = open(filepath, 'r')
        dataset = []
        pattern = re.compile('[0-9]+')
        while 1:
            line = fp.readline()
            if not line:
                break
            tokens = line.split(" ")
            words = []
            for token in tokens:
                if token in self.stopList:
                    continue
                if pattern.fullmatch(token):
                    continue
                words.append(token)
            line = " ".join(tokens)
            dataset.append(line)
        return dataset

    def jaccard_similarity_score(self, textA, textB):
        """
        Jaccard相似度计算
        :param textA:
        :param textB:
        :return:
        """
        a = textA.split(" ")
        b = textB.split(" ")
        """if len(a) > 2 and len(b) > 2:
            vec = CountVectorizer(min_df=1, ngram_range=(1, 2))
            vec.fit_transform([textA])
            a = vec.get_feature_names()
            vec.fit_transform([textB])
            b = vec.get_feature_names()"""
        c = [x for x in a if x in b]
        score = float(len(c) / (len(a) + len(b) - len(c)))
        return score

    def license_similarity(self, filepath):
        filepath = self.segments_from_file(filepath)
        dataset = self.loadset(filepath)

        X = []
        for i in dataset:
            ai = []
            for j in dataset:
                score = self.jaccard_similarity_score(i, j)
                ai.append(score)
            X.append(ai)
        np = numpy.array(X)
        print(np)



    def test(self, filepath):
        """
        测试选择最优参数
        """
        filepath = self.segments_from_file(filepath)
        dataset = self.loadset(filepath)
        self.jaccard_similarity_score(dataset[46], dataset[47])
        """
        true_ks = []
        scores = []
        for i in range(3, 200, 1):
            score = self.train(X, vectorizer, true_k=i, showLable=True) / len(dataset)
            print(i, score)
            true_ks.append(i)
            scores.append(score)
        plt.figure(figsize=(8, 4))
        plt.plot(true_ks, scores, label="error", color="red", linewidth=1)
        plt.xlabel("n_features")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        """

if __name__ == '__main__':
    license_file = './license_entities_2.txt'
    ins = LicenseCluster()
    #ins.build_word_cloud(license_file)
    ins.license_similarity(license_file)