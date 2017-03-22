# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from . import MLog


"""
A Simple Similarity Hash Class (64-bit)
Version 0.0.1
"""


class SimHash:
    def __init__(self, document):
        self.simhash = self.simhash(document)

    def __str__(self):
        return str(self.simhash)

    def simhash(self, document, topn=0):
        """
        compute sim hash
        :param document:
        :param topn: int default value 0
        :return: hash string
        """
        if topn == 0:
            topn = len(document)

        weight_vs = jieba.analyse.extract_tags(document, topK=topn, withWeight=True, allowPOS=())
        w_list = []
        for i in range(0, 64):
            w_list.append(0)

        for feature, w in weight_vs:
            hash_code = self.hash_code(feature)
            j = 0
            for i in hash_code:
                if i == '1':
                    w_list[j] += w
                else:
                    w_list[j] -= w
                j += 1
        simhash_str = ''
        for i in w_list:
            if i > 0:
                simhash_str += '1'
            else:
                simhash_str += '0'
        MLog.logger.debug('simhash: ' + simhash_str)
        return simhash_str

    @staticmethod
    def hash_code(source):
        """
        generate a 64-bit hash code from source str
        :param source:
        :return:
        """
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            return str(x)

    def hammin_dist(self, com):
        t1 = '0b' + self.simhash
        t2 = '0b' + com.simhash
        n = int(t1, 2) ^ int(t2, 2)
        i = 0
        while n:
            n &= n-1
            i += 1
        return i

if __name__ == '__main__':
    MLog.logger.debug('test case run in main method')
    simhash = SimHash("中华人民共和国公司法试行办法")
    t1 = SimHash("中华人民共和国公司法试行办法")
    t2 = SimHash("中华人民共和国公司法试行方法")
    MLog.logger.debug(simhash.hammin_dist(t1))
    MLog.logger.debug(simhash.hammin_dist(t2))
