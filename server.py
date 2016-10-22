from flask import Flask, request
from graphing import *
from pprint import pprint
app = Flask(__name__)

@app.route('/')
def hello():
    return "Welcome to the local server"

@app.route('/start')
def start():
    testrun()
    return "Initializing"

@app.route('/data', methods=['POST'])
def data():
    x = request.form['timestamp']
    ys = request.form['channel_values']
    process(3, [{'x':x, 'y' : ys[0]},{'x' : x, 'y': ys[1]},{'x' : x, 'y': ys[2]}], 0.1)
    return ""

@app.route('/end')
def end():
    stop()
    return "Stop executed"


if __name__ == "__main__":
    app.run(debug=True)