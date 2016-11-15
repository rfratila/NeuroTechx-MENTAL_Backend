from flask import Flask, request, jsonify
from flask_cors import CORS
# from graphing import *
from pprint import pprint
# from bci_workshop_tools import *
# from DeepEEG import getState
from subprocess import Popen, PIPE, STDOUT, call, check_call
import json
# import user

app = Flask(__name__)
person_name = ""
attentive = 0
time_interval = ""
CORS(app)
@app.route('/')
def hello():
    return "Welcome to the local server"

@app.route('/login', methods=['POST'])
def login():
    temp = request.get_json()
    # TODO: create a file 
    person_name = temp['name']
    return jsonify({"name" : person_name})

@app.route('/start', methods=['POST'])
def start():
    temp = request.get_json()
    attentive_state = temp['attentive']
    # attentive_state will be true / false 

    # dummy = "-p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    # args_list = dummy.split(" ")
    # p = Popen(["python", "user.py"] + args_list, stdout=PIPE, stdin=PIPE)
    # out, err = p.communicate()

    # p = Popen(["./start.sh"])
    '''
    temp = "python user.py -p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    p = Popen(temp.split(" "))
    board = user.giveBoard()
    '''
    # rc = p.poll()
    return "Initializing"

@app.route('/startFocus')
def startFocus():
    # TODO: will be used for the actual focus session
    return 

@app.route('/endFocus')
def endFocus():
    return 

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

@app.route('/lineGraphData')
def lineGraphData():
    return 

@app.route('/punchCard')
def punchCard():
    temp = {"session_start_time" : "Monday", "values": "0.3,0.1,0.2,0.7,0.4,0.6,0.33,0.9,0.8,0.1"}
    return jsonify(temp)

@app.route('/end')
def end():
    stop()
    return "Stop executed"

@app.route('/polling')
def testjson():
    # attentive has a 0 or 1 
    return jsonify({"result" : attentive })

@app.route('/registerPerson', methods=['POST'])
def registerPerson():
    temp = request.get_json()
    person_name = temp['name']
    time_interval = temp['time_interval']
    return jsonify({"name" : person_name, "time_interval" : time_interval})

@app.route('/callEEG')
def callEEG():
    # person_full_name, time_interval between samples
    getState(person_name, time_interval)

@app.route('/readFile/<name>')
def test(name):
    readFile(name)
    return "done"

def readFile(name):
    history = []
    with open(name+".txt") as f:
        for line in f:
            lst = line.split("|")
            timestamp = lst[0]
            time_interval = lst[1]
            brainStates = list(lst[2])
            history.append({"timestamp" : timestamp, "time_interval" : time_interval, "brainStates" : brainStates[:-1]})
    with open(name+".json",'w') as outfile:
        json.dump({"result":history}, outfile)
    
if __name__ == "__main__":
    app.run(debug=True)