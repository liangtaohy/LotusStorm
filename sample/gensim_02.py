from gensim import corpora, models, similarities
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Shipment of gold damaged in a fire", "Delivery of silver arrived in a silver truck", "Shipment of gold arrived in a truck"]

texts = [[word for word in document.lower().split()] for document in documents]

print(texts)

dictionary = corpora.Dictionary(texts)

print(dictionary)

print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# tfidf vector space
for doc in corpus_tfidf:
    print(doc)

print(tfidf.dfs)

print(tfidf.idfs)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

corpus_lsi = lsi[corpus_tfidf]

for doc in corpus_lsi:
    print(doc)
c = lsi[corpus]
for doc in c:
    print(doc)
index = similarities.MatrixSimilarity(lsi[corpus])

for doc in index:
    print(doc)
query = "gold silver truck"

query_bow = dictionary.doc2bow(query.lower().split())

query_lsi = lsi[query_bow]

sims = index[query_lsi]

sims = sorted(enumerate(sims), key=lambda item: -item[1])

print(list(enumerate(sims)))