# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from framework import SimHash


app = Flask(__name__)


@app.route('/simhash/generate', methods=['POST'])
def gen_simhash():
    content = request.form['document']
    simhash = SimHash.SimHash(content)
    return simhash.simhash
