#-*- coding: UTF-8 -*-
import flask
from flask import render_template, url_for
app = flask.Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    print(url_for('index'))
    return render_template('index.html')

@app.route('/month.html')
def month():
    print(url_for('month'))
    return render_template('month.html')

@app.route('/week.html')
def week():
    print(url_for('week'))
    return render_template('week.html')

@app.route('/patterns.html')
def patterns():
    print(url_for('patterns'))
    return render_template('patterns.html')

if __name__ == "__main__":
    app.run()