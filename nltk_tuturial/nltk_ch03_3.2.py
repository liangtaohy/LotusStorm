# coding=utf-8
pos = {}
pos['apple'] = 'NN'
pos['is'] = 'v'
pos['I'] = 'r'
pos['me'] = 'r'

for w in sorted(pos):
    print(w + ":" + pos[w])

print(pos.keys())
print(pos.items())

from collections import defaultdict


frequency = defaultdict(int)
frequency['colorless'] = 4
print(frequency['ideas'])

pos = defaultdict(list)
pos['sleep'] = ['NOUN', 'VERB']
print(pos['ideas'])

import nltk
from nltk.corpus import brown

tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
print(nltk.FreqDist(tags).max())

default_tagger = nltk.DefaultTagger('NN')
brown_tagged_sents = brown.tagged_sents(categories='news')
print(default_tagger.evaluate(brown_tagged_sents))

fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word,_) in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print(baseline_tagger.evaluate(brown_tagged_sents))

sent = brown.sents(categories='news')[3]
print(baseline_tagger.tag(sent))