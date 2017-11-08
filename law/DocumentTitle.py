import sys
from nltk.corpus import PlaintextCorpusReader
import nltk, re, pprint
from nltk import word_tokenize
from law import JiebaTokenizer

print(sys.getdefaultencoding())

corpus_root = '/Users/xlegal/PycharmProjects/LotusStorm/sample'
tokenizer = JiebaTokenizer('../framework/stop_words_jieba.utf8.txt')
wordlist = PlaintextCorpusReader(corpus_root, 'titles.txt', word_tokenizer=tokenizer)
words = wordlist.words()

s = [word for (word, flag) in words if word == '关于']
fdist = nltk.FreqDist(w for w in s)
print(fdist.most_common(100))
#fdist = nltk.FreqDist(flag for (word, flag) in words)
#print(fdist.most_common(100))
fdist.plot(100, cumulative=False)