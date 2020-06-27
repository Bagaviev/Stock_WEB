#-*- coding: UTF-8 -*-
import flask, pandas
from flask import render_template, url_for

data = pandas.read_csv('static\data\parsed_EURUSD_y.csv', delimiter=';')
act = data.is_actual.tail(1).to_string(index=False)
cls = data.last_CLOSE.tail(1).to_string(index=False)

app = flask.Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    print(url_for('index'))
    return render_template('index.html', act=act, cls=cls)

@app.route('/month.html')
def month():
    print(url_for('month'))
    return render_template('month.html', act=act, cls=cls)

@app.route('/week.html')
def week():
    print(url_for('week'))
    return render_template('week.html', act=act, cls=cls)

@app.route('/patterns.html')
def patterns():
    print(url_for('patterns'))
    return render_template('patterns.html', act=act, cls=cls)

if __name__ == "__main__":
    app.run()