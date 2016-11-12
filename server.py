from flask import Flask, request, jsonify
from flask_cors import CORS
# from graphing import *
from pprint import pprint
# from bci_workshop_tools import *
# from DeepEEG import getState
from subprocess import Popen, PIPE, STDOUT, call, check_call
# import user

app = Flask(__name__)
person_name = ""
time_interval = ""
CORS(app)
@app.route('/')
def hello():
    return "Welcome to the local server"

@app.route('/start')
def start():

    # dummy = "-p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    # args_list = dummy.split(" ")
    # p = Popen(["python", "user.py"] + args_list, stdout=PIPE, stdin=PIPE)
    # out, err = p.communicate()

    # p = Popen(["./start.sh"])
    temp = "python user.py -p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    p = Popen(temp.split(" "))
    board = user.giveBoard()

    # rc = p.poll()
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
    temp = request.get_json()
    person_name = temp['name']
    time_interval = temp['time_interval']
    pprint("person_name is " + person_name)
    pprint("time_interval is "+ time_interval)
    return jsonify({"name" : person_name, "time_interval" : time_interval})

@app.route('/callEEG')
def callEEG():
    # person_full_name, time_interval between samples
    getState(person_name, time_interval)

if __name__ == "__main__":
    app.run(debug=True)