import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

print("ebal rti")

if __name__ == "__main__":
    app.run()