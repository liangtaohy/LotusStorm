# -*- coding: utf-8 -*-

import json
from flask import Flask
from flask import request
from framework import SimHash
from framework import Extractor


app = Flask(__name__)

simhash_dict = "/mnt/open-xdp/LotusStorm/simhash_dict.txt"


@app.route('/simhash/generate', methods=['POST'])
def gen_simhash():
    content = request.form['document']
    simhash = SimHash.SimHash(content)

    repeated = 0

    dict = open(simhash_dict)

    try:
        for line in dict:
            line = line.strip(' \r\t\n')
            if line:
                dist = simhash.hammin_dist_simple(line)
                if dist <= 3:
                    repeated = 1
                    break
    finally:
        dict.close()

    if repeated == 0:
        dict = open(simhash_dict, 'w')
        dict.write(simhash.simhash + "\n")
        dict.close()

    return json.dumps({'simhash':simhash.simhash, 'repeated':repeated}, ensure_ascii=False)


@app.route('/extractor/contents', methods=['POST'])
def extract_contents():
    document = request.form['document']
    return json.dumps(Extractor.Extractor().extract_contents(document), ensure_ascii=False)


@app.route('/extractor/gen/semantic_web', methods=['POST', 'GET'])
def extract_gen_semantic_web():
    id = request.form['id']
    type = request.form['type']
    #