#-*- coding: UTF-8 -*-
import flask
from flask import render_template, flash, redirect, url_for, request
app = flask.Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    print(url_for('index'))
    return render_template('index.html')

if __name__ == "__main__":
    app.run()