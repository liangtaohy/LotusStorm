# -*- coding: utf-8 -*-

import json
from flask import Flask
from flask import request
from framework import SimHash
from framework import Extractor


app = Flask(__name__)


@app.route('/simhash/generate', methods=['POST'])
def gen_simhash():
    content = request.form['document']
    simhash = SimHash.SimHash(content)
    return simhash.simhash


@app.route('/extractor/contents', methods=['POST'])
def extract_contents():
    document = request.form['document']
    return json.dumps(Extractor.Extractor().extract_contents(document), ensure_ascii=False)


@app.route('/extractor/gen/semantic_web', methods=['POST', 'GET'])
def extract_gen_semantic_web():
    id = request.form['id']
    type = request.form['type']
    #