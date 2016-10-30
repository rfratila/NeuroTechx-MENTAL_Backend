from flask import Flask, request, jsonify
from graphing import *
from pprint import pprint
from bci_workshop_tools import *

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
    x = request.form['sample_number']
    ys = request.form['channel_values']
    delay = request.form['delay']
    li = []
    for y in ys:
        li.append({"x" : x, 'y' : y})
    process(8, li, delay)
    return ""

@app.route('/end')
def end():
    stop()
    return "Stop executed"

@app.route('/testjson')
def testjson():
    return jsonify({"first" : [1,2,3,4], "second" : {"test1" : "test2"}})

@app.route('/recording1')
def recording():
    

if __name__ == "__main__":
    app.run(debug=True)