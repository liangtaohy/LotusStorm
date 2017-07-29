# coding=utf-8

from __future__ import division  # Python 2 users only
from nltk.corpus import PlaintextCorpusReader
import nltk, re, pprint
from nltk import word_tokenize
from text_proc import *
from urllib import request
from bs4 import BeautifulSoup
import json
import jieba


corpus_root = '/Users/xlegal/nltk_dict/docs'
wordlists = PlaintextCorpusReader(corpus_root, '7e648826423237a1a9d548c07303c033.txt')
print(wordlists.fileids())
for w in wordlists.raw('7e648826423237a1a9d548c07303c033.txt')[:100]:
    print(w,' ')

f = open(corpus_root + '/7e648826423237a1a9d548c07303c033.txt')

raw = f.read()

jobj = json.loads(raw)
print(jobj['title'])
tokens = word_tokenize(jobj['content'])
print(tokens[:20])
text = nltk.Text(tokens)
print(text[:20])

tset = set(jobj['content'])
print(tset)
text = nltk.Text(tset)
print(text[:20])

seg_list = jieba.cut(jobj['content'], cut_all=False)
print(type(seg_list))
tokens = word_tokenize(' '.join(seg_list))
print(tokens[:20])
print(set(tokens))
print(lexical_diversity(tokens))
fd = nltk.FreqDist(tokens)
print(fd)
print(fd.most_common(10))