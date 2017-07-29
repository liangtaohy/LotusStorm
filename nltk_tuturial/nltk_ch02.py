import nltk
from nltk.corpus import gutenberg

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars / num_words),round(num_words / num_sents),round(num_words / num_vocab),fileid)

macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')

from nltk.corpus import brown

news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
modal = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modal:
    print(m + ':', fdist[m], end=' ')

cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)