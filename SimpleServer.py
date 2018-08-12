# -*- coding: utf-8 -*-

import os
from flask import Flask
from flask import request, redirect, url_for
from framework import SimHash
from framework import Extractor
from framework import predict
from law import DocumentParser
import pymysql
from law import Settings as settings
import random
from werkzeug.utils import secure_filename
from corpus.ts import TsDocx
from corpus.ts import TsTermTokens
import json
import pickle
import jieba

label_ = json.load(open("./corpus/samples/ts/labels.json", "r", encoding="utf-8"))

cls_file = 'tf/bayes/.bayes_classifier.pickle'
word_bag_file = 'tf/bayes/.word_bag.json'

cls = pickle.load(open(cls_file, "rb"))

word_bag = json.load(open(word_bag_file, "r", encoding="utf-8"))

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = ['docx']
ALLOWED_MIMES = ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

simhash_dict = "./simhash_dict.txt"

def stop_words_local():
    stop_words_file = open(os.path.join(os.path.dirname(__file__), 'tf/bayes/stop_words_utf8.txt'), 'r',
                           encoding='utf-8')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line[:-1])

    if os.path.exists("./unrelative_words.json"):
        words = json.load(open("./unrelative_words.json", 'r', encoding="utf-8"))
        stopwords_list.extend(words)
    stopwords_list.extend(["满意","唆使"])
    return list(set(stopwords_list))


def jieba_fenci(raw, stopwords_list):
    word_list = list(jieba.cut(raw))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    word_list = [word for word in word_list if word not in stopwords_list]
    for word in word_list:
        if word in ['•', '轮轮', '定义']:
            word_list.remove(word)
    """if '\n' in word_list:
        word_list.remove('\n')"""
    return word_list


def document_features(data, word_bag):
    """
    labeled_featuresets: A list of classified featuresets,
    i.e., a list of tuples ``(featureset, label)``.
    """
    feature = {}
    tokens = set(data)
    for w in word_bag:
        if w in tokens:
            feature[w] = 1
        else:
            feature[w] = 0
    return feature


def bayes_prob(cls, word_bag, input_text):
    """
    贝叶斯分类预测
    :param cls_file:
    :param word_bag_file:
    :param input_text:
    :return:
    """
    input_set = jieba_fenci(input_text, stopwords_list=stop_words_local())
    input_feature = document_features(input_set, word_bag)
    result = cls.prob_classify(input_feature)
    return result.max()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return json.dumps({'code': 4, 'message': '请选择您要上传的文件'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return json.dumps({'code': 3, 'message': "文件名不可为空"})
        if file and allowed_file(file.filename) and file.mimetype in ALLOWED_MIMES:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            wps, lines, tr_total, wp_total = TsDocx.guess_wp_type(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            doc = TsDocx.handle_wps(wps, term_dict=TsTermTokens.TsTermDict)
            invalid_char = ['%', ':', '.', '。', '，', '；', ';']

            for i in range(len(doc['terms'])):
                term = doc['terms'][i]
                s = term['term']
                if term['p'] < 0.3:
                    label = 0
                else:
                    if s.strip()[-1] in invalid_char:
                        label = 0
                    else:
                        label = bayes_prob(cls, word_bag,
                                           input_text="{0},{1}".format(term['term'], term['content']))

                if label > 0:
                    for k, v in label_.items():
                        if label == v:
                            doc['terms'][i]['label'] = k
                            break
                else:
                    doc['terms'][i]['label'] = '未识别'

            #json.dump({'code': 0, 'doc': doc}, open("./uploads/%s.json" % file.filename, 'w', encoding='utf-8'),
            #          ensure_ascii=False)
            return json.dumps({'code': 0, 'doc': doc}, ensure_ascii=False)
        else:
            return json.dumps({'code': 1, 'message': "仅支持%s等类型的文档" % (",".join(ALLOWED_EXTENSIONS))})
    else:
        return json.dumps({'code': 2, 'message': "文件上传仅支持POST请求!!"})


@app.route('/bayesclassifier/lawtitle2tag', methods=['POST'])
def predict_tag():
    title = request.form['title']
    topClazz = predict.predictTag(title)
    res = {'code': 0, 'content': {'tags': topClazz}}
    return json.dumps(res, ensure_ascii=False)


@app.route('/simhash/generate', methods=['POST'])
def gen_simhash():
    content = request.form['document']
    simhash = SimHash.SimHash(content)

    repeated = 0
    simhash1 = ''

    dict = open(simhash_dict)

    try:
        for line in dict:
            line = line.strip(' \r\t\n')
            if line:
                dist = simhash.hammin_dist_simple(line)
                if dist <= 3:
                    repeated = 1
                    simhash1 = line
                    break
    finally:
        dict.close()

    if repeated == 0:
        dict = open(simhash_dict, 'a')
        dict.write(simhash.simhash + "\n")
        dict.close()

    res = {'code': 0, 'content': {'simhash': simhash.simhash, 'repeated': repeated, 'simhash1': simhash1}}

    return json.dumps(res, ensure_ascii=False)


@app.route('/extractor/contents', methods=['POST'])
def extract_contents():
    document = request.form['document']
    return json.dumps(Extractor.Extractor().extract_contents(document), ensure_ascii=False)


@app.route('/extractor/gen/semantic_web', methods=['POST', 'GET'])
def extract_gen_semantic_web():
    id = request.form['id']
    type = request.form['type']
    return ""


@app.route('/text/tokens', methods=['POST', 'GET'])
def text_tokens():
    stop_word = request.args.get('stop_word', False)
    content = request.args.get('content', False)

    document = {
        'author': request.args.get('author', ''),
        'title': request.args.get('text', '欢迎使用'),
    }
    parser = DocumentParser.DocumentParser(document, stop_word)
    parser.set_user_dict(os.path.dirname(os.path.realpath(__file__)) + '/framework/user_dict.txt')
    parser.parse_title()

    if content and len(content):
        parser.parse_relations(content)
    tokens = [({'word': word, 'postag': flag, 'postag_str': ''}) for (word, flag) in parser.tokens]

    res = {'code': 0, 'content': {'entity': parser.entity, 'tokens': tokens}}

    return json.dumps(res, ensure_ascii=False)


@app.route("/citation", methods=['GET'])
def citation():
    doc_id = request.args.get('doc_id', 0)
    db = pymysql.connect(
        host=settings.MYSQL_HOST,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASS,
        db=settings.MYSQL_DB,
        charset=settings.CHARSET,
        cursorclass=pymysql.cursors.DictCursor
    )

    table = "document_citations"

    cursor = db.cursor()

    cursor.execute("SELECT * FROM " + table + " WHERE doc_id=" + doc_id)

    cites = cursor.fetchall()

    nodes = [
        {
            "id": 'doc_id_' + doc_id,
            "label": doc_id,
            "x": 4,
            "y": 4,
            "size": 2,
            "color": 'green'
        },
    ]

    edges = []

    for cite in cites:
        node = {
            "id": 'id_%d' % cite['id'],
            "label": cite['cite_doc_title'],
            "x": random.random(),
            "y": random.random(),
            "size": 1,
            "color": 'red'
        }
        nodes.append(node)

        edge = {
            "id": 'e_%d_%d' % (int(doc_id), cite['id']),
            "source": 'doc_id_' + doc_id,
            "target": 'id_%d' % cite['id'],
            "size": random.random(),
            "color": '#ccc'
        }

        edges.append(edge)

    ret = {
        "nodes": nodes,
        "edges": edges,
    }

    return json.dumps(ret, ensure_ascii=False), 200, {'Content-Type': 'application/json; charset=utf-8'}


app.run(debug=True, host='0.0.0.0', port=8080)
