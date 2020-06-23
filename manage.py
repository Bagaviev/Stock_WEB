import flask

app = flask.Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

print("ebal rti vseh")

if __name__ == "__main__":
    app.run()