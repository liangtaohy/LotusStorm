# coding=utf-8

import codecs, os, operator, jieba, string, datetime, random, re, pickle
from numpy import zeros, ones, log

#from bayes import loadDataSet, testNaiveBayes

stopwords = [u"第", u"已", u"号", u"、", u"的", u"上", u"反", u"后", u"》", u"《", u"其", u"谈", u"蒙", u"第一", u"第二", u"第三", u"第五", u"不",  u"第七", u"第九"]

modelFile = None
dumpObj = None
topics, words, matrix, topicsProb = None, None, None, None

def predictTag(title):
    global modelFile, dumpObj, topics, words, matrix, topicsProb
    if modelFile == None:
        modelFile = open(os.path.join(os.path.dirname(__file__), 'models_average.dump'), 'rb')
        dumpObj = pickle.load(modelFile, encoding='bytes')
        topics, words, matrix, topicsProb = dumpObj[b'topics'], dumpObj[b'words'], dumpObj[b'matrix'], dumpObj[b'topicsProb']
    N = 3

    numOfTopics = len(topics)
    numOfWords = len(words)
    vec = zeros(numOfWords)

    for token in jieba.cut_for_search(title):
        if token not in stopwords and token in words:
            vec[words.index(token)] += 1

    clazz = zeros(numOfTopics)
    for j in range(numOfTopics):
        clazz[j] = sum(vec * matrix[j]) # + log(topicsProb[j])
        
    idxClazz = [(k, clazz[k]) for k in range(numOfTopics)]
    classSorted = sorted(idxClazz, key=operator.itemgetter(1), reverse=True)[:N]
    topClazz = [topics[t] for t, cz in classSorted]
    ret = []
    for i in range(len(topClazz)):
        ret.append(topClazz[i].decode('utf-8').strip('\n'))
    return ret


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

        print('预测语句标签', item)
        for i in range(len(topClazz)):
            print(topClazz[i].decode('utf-8').strip('\n'))


sentences = [u"中央军委关于总参军事气象局与中央气象局合并问题的通知", u"新西兰政府和中华人民共和国政府关于对所得避免双重征税和防止偷漏税的协定的第二个议定书", u"厦门市人民政府关于修改《厦门市城镇职工基本医疗保险规定》的决定", u"珠海市妇女联合会关于表彰珠海市三八红旗集体三八红旗手的决定", u"会计师事务所新旧财务、会计制度衔接办法［失效］", u"中华人民共和国保守国家秘密法实施条例,", u"第五届全国人民代表大会第三次会议关于修改《中华人民共和国宪法》第四十五条的决议［失效］", u"中华人民共和国民用航空法（2015年修正）", u"国务院办公厅关于深化改革推进出租汽车行业健康发展的指导意见", u"建设部、民政部、全国老龄办、中国残联关于表彰无障碍设施建设先进区的决定", u"共青团中央、教育部关于中学共青团工作几个具体问题的规定", u"国务院关于国家经委成立全国饲料工业机构问题的批复", u"国务院、中央军委关于一九八五年招收飞行学员的通知", u"国务院关于发给离休退休人员生活补贴费的通知"]


def test():
    modelFile = open(os.path.join(os.path.dirname(__file__), 'models_average.dump'), 'rb')
    dumpObj = pickle.load(modelFile, encoding='bytes')
    predict(sentences, dumpObj[b'topics'], dumpObj[b'words'], dumpObj[b'matrix'], dumpObj[b'topicsProb'], 3)

"""
def totalTest():
    modelFile = open(os.path.join(os.path.dirname(__file__), 'models_average.dump'), 'r')
    dumpObj = pickle.load(modelFile, encoding='bytes')
    topics, words, matrix, topicsProb = dumpObj[b'topics'], dumpObj[b'words'], dumpObj[b'matrix'], dumpObj[b'topicsProb']
    dataSet = loadDataSet('ds.csv')
    N = 3
    testNaiveBayes(dataSet, topics, words, matrix, topicsProb, N)
"""

if __name__ == '__main__':
    modelFile = open(os.path.join(os.path.dirname(__file__), 'models_average.dump'), 'rb')
    dumpObj = pickle.load(modelFile, encoding='bytes')
    topics, words, matrix, topicsProb = dumpObj[b'topics'], dumpObj[b'words'], dumpObj[b'matrix'], dumpObj[b'topicsProb']
    while True:
        sentence = input("输入待分类文件title，结束请输入exit:\n")
        if sentence == "exit":
            break
        else:
            predict([sentence], topics, words, matrix, topicsProb, 3)
    
