import sys
sys.path.append("../")

from utils import tfidf


if __name__ == "__main__":
    file = sys.argv[1]

    f = open(file, "r", encoding='utf-8')
    lines = []

    while 1:
        l = f.readline()
        if not l:
            break
        lines.append(l)

    word, tfidf_mat = tfidf(lines)

    weight = tfidf_mat.toarray()

    tokens = []
    for i in range(len(weight)):
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")

        l = []

        for j in range(len(word)):
            if weight[i][j]:
                print(word[j], weight[i][j])
                l.append("%s,%f" % (word[j], weight[i][j]))
        tokens.append(" ".join(l))

    print("\n".join(tokens))
