# -*- coding: utf-8 -*-

import os
import json
from flask import Flask
from flask import request
from framework import SimHash
from framework import Extractor
from framework import predict
from law import DocumentParser
import pymysql
from law import Settings as settings
import random
from law.JiebaPosTagsDict import PosTagDict


app = Flask(__name__)

simhash_dict = "./simhash_dict.txt"


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


app.run(debug=True, host='0.0.0.0', port=5000)
