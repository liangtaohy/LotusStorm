# -*- coding: utf-8 -*-

import json
from flask import Flask
from flask import request
from framework import SimHash
from framework import Extractor
from framework import predict


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
    #
