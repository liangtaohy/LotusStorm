import os
import sys
sys.path.append("../")
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

from utils.CorpusTfIDF import tfidf

from framework.JiebaTokenizer import JiebaTokenizer
jieba = JiebaTokenizer(stop_words_path=os.path.dirname(os.path.realpath(__file__)) +
                                                    '/../framework/stop_words_jieba.utf8.txt')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import (manifold, datasets, decomposition, ensemble,random_projection)


def tokenize(content):
    tokens = jieba.tokenize(content)
    segs = [w for (w, _) in tokens]
    return segs


# 将降维后的数据可视化,2维
def plot_embedding_2d(X, title=None, targets=None):
    #坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    #降维后的坐标为（X[i, 0], X[i, 1]），在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(targets[i]),
                 color=plt.cm.Set1(targets[i]/10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]


    user_dict = None

    if len(sys.argv) >= 4:
        user_dict = sys.argv[3]

    if user_dict:
        jieba.set_user_dict(user_dict_file=user_dict)

    f = open(input, "r", encoding='utf-8')
    lines = []

    while 1:
        l = f.readline()
        if not l:
            break
        tokens = tokenize(l)
        lines.append(" ".join(tokens) + "\n")

    f.close()

    f = open(output, "w", encoding='utf-8')
    f.writelines(lines)
    f.close()

    t0 = time()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(lines)
    names = vectorizer.get_feature_names()
    """print(X)
    print(names)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()"""

    words, tfidf_matrix = tfidf(lines)
    #print("total words: %d" % len(words))
    print(words)
    print(tfidf_matrix)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=20, max_iter=300, n_init=1, verbose=True)
    t0 = time()
    kmeans.fit(X)
    """print("done in %0.3fs" % (time() - t0))
    print()
    print(kmeans.labels_)
    print("total labels: %d" % len(kmeans.labels_))
    print(kmeans.cluster_centers_)
    print("center count: %d" % len(kmeans.cluster_centers_))"""
    print(kmeans.labels_)
    #for center in kmeans.cluster_centers_:
    #    print(center)
    for i in range(len(kmeans.labels_)):
        print("%d\t%s" % (kmeans.labels_[i], lines[i].strip()))

    # lle required parameters
    n_neighbors = 20

    # PCA
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding_2d(X_pca[:, 0:2], "PCA 2D", kmeans.labels_)

    # modified LLE
    print("Computing modified LLE embedding")
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
    t0 = time()
    X_mlle = clf.fit_transform(X.toarray())
    print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
    plot_embedding_2d(X_mlle, "Modified Locally Linear Embedding (time %.2fs)" % (time() - t0), kmeans.labels_)

    # MDS
    print("Computing MDS embedding")
    clf = manifold.MDS(n_components=2, n_init=1, max_iter=300)
    t0 = time()
    X_mds = clf.fit_transform(X.toarray())
    print("Done. Stress: %f" % clf.stress_)
    plot_embedding_2d(X_mds, "MDS (time %.2fs)" % (time() - t0), kmeans.labels_)

    # t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X.toarray())
    print(X_tsne.shape)
    plot_embedding_2d(X_tsne[:, 0:2], "t-SNE 2D", kmeans.labels_)

    plt.show()