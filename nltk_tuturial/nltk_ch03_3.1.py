# coding=utf-8
from __future__ import division  # Python 2 users only
import nltk, re, pprint
from nltk import word_tokenize
from urllib import request
from bs4 import BeautifulSoup


url = "http://www.gutenberg.org/files/2554/2554.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
print(type(raw))
print(len(raw))
print(raw[:75])

tokens = word_tokenize(raw)

print(type(tokens))
print(len(tokens))
print(tokens[:10])

text = nltk.Text(tokens)
print(type(text))
print(text[1024:1062])
text.collocations()

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = request.urlopen(url).read().decode('utf8')
print(html[:60])

raw = BeautifulSoup(html, "html.parser").get_text()
tokens = word_tokenize(raw)
print(tokens[:10])

url = "http://www.chinacourt.org/law/detail/2005/10/id/104490.shtml"
html = request.urlopen(url).read().decode('utf8')
print(html[:60])
raw = BeautifulSoup(html, "html.parser").get_text()
tokens = word_tokenize(raw)
print(tokens[:10])