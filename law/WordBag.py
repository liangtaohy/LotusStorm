import os
import pymysql
import math
import time
import re
from wordcloud import WordCloud
import PIL
import numpy
import matplotlib.pyplot as plt
from law import JiebaTokenizer
from law import DocType
from law import GovDict
from law import Settings as settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans


class DocClassifierByTitles:

    def __init__(self):
        self.db = pymysql.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASS,
            db=settings.MYSQL_DB,
            charset=settings.CHARSET,
            cursorclass=pymysql.cursors.DictCursor
        )

        self.table = 'gov'

        self.cursor = self.db.cursor()
        self.tokenizer = JiebaTokenizer.JiebaTokenizer(os.path.dirname(os.path.realpath(__file__)) +
                                                  '/../framework/stop_words_jieba.utf8.txt')

        self.stopList = [line.strip() for line in open(os.path.dirname(os.path.realpath(__file__)) +
                                                  '/../framework/stop_words_jieba.utf8.txt')]

        self.stopList = self.stopList + [' ', '\n', '\t']

        print('db open')

    def total(self):
        self.cursor.execute("SELECT COUNT(*) AS total FROM %s" % self.table)
        data = self.cursor.fetchone()
        return data['total']

    def loaddataintorawfile(self):
        total = self.total()
        pagesize = 1000
        pages = math.ceil(total / pagesize)
        if not os.path.exists('./law_raw.txt'):
            fp = open('./law_raw.txt', 'a')
            for page in range(pages):
                self.cursor.execute(
                    "SELECT title FROM " + self.table + " WHERE 1 LIMIT %d,%d" % (page * pagesize, pagesize))
                rows = self.cursor.fetchall()
                ts = []
                for row in rows:
                    title = row['title'].strip()
                    if len(title):
                        ts.append(title + '\n')
                fp.writelines(ts)
            fp.close()

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

    def parse_lines(self, file):
        """
        parse file to tokens
        :param file:
        :return:
        """
        fp = open(file)
        output_file = open('./title_segs.txt', 'w')
        while 1:
            line = fp.readline()
            if not line:
                break
            tokens = self.tokenizer.tokenize(line)
            words = []
            for (word, flag) in tokens:
                w = word.strip()
                if len(w):
                    words.append(word)
            if len(words):
                output_file.write(" ".join(words) + "\n")

        fp.close()
        output_file.close()

    def buildWordCloud(self):
        """
        build WordCloud Image
        :return:
        """
        text = open('./title_segs.txt').read()
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
                for ind in order_centroids[i, :10]:
                    print(' %s' % terms[ind], end='')
                print()
        return -km.score(X)

    def doc_cites(self):
        total = self.total()
        pagesize = 1000
        pages = math.ceil(total / pagesize)
        output_to = './doc_cite_entity.txt'
        if not os.path.exists(output_to):
            fp = open(output_to, 'a')
            for page in range(pages):
                self.cursor.execute(
                    "SELECT cite_doc_title FROM document_citations " + " WHERE 1 LIMIT %d,%d" % (page * pagesize, pagesize))
                rows = self.cursor.fetchall()
                ts = []
                for row in rows:
                    title = row['cite_doc_title'].strip()
                    title = title.replace('\n', '')
                    if len(title):
                        ts.append(title + '\n')
                fp.writelines(ts)
            fp.close()

    def test(self, filepath):
        """
        测试选择最优参数
        """
        dataset = self.loadset(filepath)
        print("%d documents" % len(dataset))
        X, vectorizer = self.tfidf(dataset, n_features=500)
        true_ks = []
        scores = []
        for i in range(3, 80, 1):
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

if __name__ == '__main__':
    print("cls begin")
    cls = DocClassifierByTitles()
    begin = (int(round(time.time() * 1000)))
    #cls.loaddataintorawfile()
    #cls.parse_lines('./law_raw.txt')
    #cls.buildWordCloud()
    #cls.test('./title_segs.txt')
    cls.doc_cites()
    end = (int(round(time.time() * 1000)))
    print("Time used: %d" % (end-begin))
