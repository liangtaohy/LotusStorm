# coding=utf-8

import codecs, os, operator, jieba, string, datetime, random, re, pickle

from numpy import zeros, ones, log

# identify = string.maketrans("", "")
stopwords = [u"第", u"已", u"号", u"、", u"的", u"上", u"反", u"后", u"》", u"《", u"其", u"谈", u"蒙", u"第一", u"第二", u"第三", u"第五", u"不",  u"第七", u"第九"]

def splitTrainTest(total, seed, M, k):
    random.seed(seed)
    train = []
    test = []
    for line in total:
        if random.randint(0, M - 1) == k:
            test.append(line)
        else:
            train.append(line)
    return train, test

def loadDataSet(filename):
    ret = []
    i = 0
    for line in open(filename).readlines():
        """
        if i > 10000:
            break
        i += 1
        """

        tmp = line.split(',')
        title, _ = re.subn(r'[()0-9]', '', tmp[1])
        topic = tmp[2]

        tokens = []
        # for token in jieba.cut(title, cut_all=False):
        for token in jieba.cut_for_search(title):
            if token not in stopwords:
                tokens.append(token)

        ret.append([tokens, topic])
    return ret

def statDataSet(dataSet):
    topicDict = {}
    tokenSet = set()

    topic_total = len(dataSet)

    for item in dataSet:
        topic = item[1]
        topicDict[topic] = topicDict.get(topic, 0) + 1
        for token in item[0]:
            tokenSet.add(token)

    # topicSorted = sorted(topicDict.items(), key=operator.itemgetter(1), reverse=True)

    topics = []
    topicsProb = []

    for topic, num in topicDict.items():
        topics.append(topic)
        topicsProb.append(num / float(topic_total))

    words = list(tokenSet)

    return topics, words, topicsProb

def trainNaiveBayes(train, topics, words):
    numOfDocs = len(train)
    numOfTopics = len(topics)
    numOfWords = len(words)

    matrix = ones((numOfTopics, numOfWords))

    for doc in train:
        vec = zeros(numOfWords)
        for token in doc[0]:
            if token in words:
                vec[words.index(token)] += 1

        topic = doc[1]
        if topic in topics:
            topicIdx = topics.index(topic)
            matrix[topicIdx] += vec

    for tIdx in range(numOfTopics):
        matrix[tIdx] = log(matrix[tIdx] / float(sum(matrix[tIdx])))

    return matrix

def testNaiveBayes(test, topics, words, matrix, priori, N=1):
    numOfDocs = len(test)
    numOfTopics = len(topics)
    numOfWords = len(words)

    numOfCorrect = zeros(N)

    for i in range(numOfDocs):
        doc = test[i]

        vec = zeros(numOfWords)
        for token in doc[0]:
            if token in words:
                vec[words.index(token)] += 1

        clazz = zeros(numOfTopics)
        for j in range(numOfTopics):
            clazz[j] = sum(vec * matrix[j]) # + log(priori[j])
        
        idxClazz = [(k, clazz[k]) for k in range(numOfTopics)]
        classSorted = sorted(idxClazz, key=operator.itemgetter(1), reverse=True)[:N]

        topClazz = [topics[t] for t, cz in classSorted]
        for n in range(N):
            for m in range(n + 1):
                if doc[1] == topClazz[m]:
                    numOfCorrect[n] += 1
                    break

    numOfCorrect = numOfCorrect / float(numOfDocs)

    for i in range(N):
        print("Top %d候选覆盖命中正确率:%f" % (i + 1, numOfCorrect[i]))
    """
    """

    return numOfCorrect

def predict(data, topics, words, matrix, priori, N=1):
    numOfTopics = len(topics)
    numOfWords = len(words)

    for item in data:
        vec = zeros(numOfWords)

        for token in jieba.cut_for_search(item):
            if token not in stopwords and token in words:
                vec[words.index(token)] += 1

        clazz = zeros(numOfTopics)
        for j in range(numOfTopics):
            clazz[j] = sum(vec * matrix[j]) # + log(priori[j])
        
        idxClazz = [(k, clazz[k]) for k in range(numOfTopics)]
        classSorted = sorted(idxClazz, key=operator.itemgetter(1), reverse=True)[:N]
        topClazz = [topics[t] for t, cz in classSorted]

        print('预测语句%s的标签如下:' % item)
        print(','.join(topClazz))


sentences = [u"中央军委关于总参军事气象局与中央气象局合并问题的通知", u"新西兰政府和中华人民共和国政府关于对所得避免双重征税和防止偷漏税的协定的第二个议定书", u"厦门市人民政府关于修改《厦门市城镇职工基本医疗保险规定》的决定", u"珠海市妇女联合会关于表彰珠海市三八红旗集体三八红旗手的决定", u"会计师事务所新旧财务、会计制度衔接办法［失效］", u"中华人民共和国保守国家秘密法实施条例,", u"第五届全国人民代表大会第三次会议关于修改《中华人民共和国宪法》第四十五条的决议［失效］", u"中华人民共和国民用航空法（2015年修正）", u"国务院办公厅关于深化改革推进出租汽车行业健康发展的指导意见", u"建设部、民政部、全国老龄办、中国残联关于表彰无障碍设施建设先进区的决定", u"共青团中央、教育部关于中学共青团工作几个具体问题的规定", u"国务院关于国家经委成立全国饲料工业机构问题的批复", u"国务院、中央军委关于一九八五年招收飞行学员的通知", u"国务院关于发给离休退休人员生活补贴费的通知"]


if __name__ == '__main__':
    print('start')

    dataSet = loadDataSet('ds.csv')

    topics, words, topicsProb = statDataSet(dataSet)

    print('numOfTopics:%d, numOfWords:%d' % (len(topics), len(words)))

    """
    print topicsProb
    wordFile = open("words", "w")
    content = "\t".join(words)
    wordFile.write(content.encode("utf-8"))
    wordFile.close()
    """

    M = 16
    N = 3
    cs = []
    seed = datetime.datetime.now()

    total_matrix = zeros((len(topics), len(words)))

    for i in range(M):
        """
        if i > 0:
            break
        """
        train, test = splitTrainTest(dataSet, seed, M, i)
        matrix = trainNaiveBayes(train, topics, words)
        cs.append(testNaiveBayes(test, topics, words, matrix, topicsProb, N))
        total_matrix = total_matrix + matrix

    total_matrix = total_matrix / M

    dumpObj = {'topics': topics, 'words': words, 'topicsProb': topicsProb, 'matrix': total_matrix}

    modelFile = open('models_average.dump', 'w')
    pickle.dump(dumpObj, modelFile)

    predict(sentences, topics, words, matrix, topicsProb, 3)

    for i in range(N):
        sum = 0
        for j in range(M):
            sum += cs[j][i]
        print("Top %d候选平均覆盖命中正确率:%f" % (i + 1, sum / float(M)))

    print('end')


