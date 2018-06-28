from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def tfidf(corpus):
    """
    Get bag-of-word and tfidf matrix (i is document[i], j is word[j] in bag of word)
    :param corpus:
    :return:
    """
    vectorizer = CountVectorizer()  # TF Matrix
    transformer = TfidfTransformer()  # TFIDF Transformer
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    """
    for i in range(len(weight)):
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])
    """

    return word, tfidf
