# -*- coding: utf-8 -*-
from framework import MLog
import gensim
from gensim.models import word2vec

model = gensim.models.Word2Vec.load("wensu.case.model")

result = model.most_similar("判决书")

for e in result:
    print(e[0], e[1])
