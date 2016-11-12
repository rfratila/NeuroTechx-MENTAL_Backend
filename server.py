from flask import Flask, request, jsonify
from graphing import *
from pprint import pprint
from bci_workshop_tools import *
# from DeepEEG import getState
import subprocess

app = Flask(__name__)
person_name = ""
time_interval = ""

@app.route('/')
def hello():
    return "Welcome to the local server"

@app.route('/start')
def start():
    subprocess.call("test.sh")
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

@app.route('/registerPerson', methods=['POST'])
def registerPerson():
    pprint(request)
    pprint(request.args)
    pprint(request.form)
    person_name = request.form['name']
    time_interval = request.form['time_interval']
    pprint("person_name is " + person_name)
    pprint("time_interval is "+ time_interval)
    return jsonify({"name" : person_name, "time_interval" : time_interval})

@app.route('/callEEG')
def callEEG():
    # person_full_name, time_interval between samples
    getState(person_name, time_interval)

if __name__ == "__main__":
    app.run(debug=True)