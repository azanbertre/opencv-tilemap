from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/get_json')
def get_json():
    return ""

if __name__ == '__main__':
    app.run()