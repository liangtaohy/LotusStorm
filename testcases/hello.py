# -*- coding: utf-8 -*-

from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/')
def index():
    return 'Lotus Index Page'


@app.route("/hello")
def hello():
    return "hello,world!"


@app.route("/user/<username>")
def userinfo(username):
    return "Your Name is: " + username


@app.route("/post/<int:posid>")
def user_post_id(posid):
    return "You request post id: " + format(posid)


@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        return "Your Name " + request.form['username'] + ", password " + request.form['password']
    else:
        return "Please login"


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.txt')