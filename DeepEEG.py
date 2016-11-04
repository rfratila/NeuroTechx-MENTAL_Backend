import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os

from collections import OrderedDict
import pylab #for graphing

import json

#TODO:
#have a fully convolutional layer to account for variable recording time
#save some samples for testing and validation

# Custom softmax function to support fully convolutional networks
def softmax(x):
	e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
	return e_x / e_x.sum(axis=1, keepdims=True)


#inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%dataPath)# got this from: http://bnci-horizon-2020.eu/database/data-sets
def getData(dataPath):

	trainingSet =[]
	counter = 0
	for patient in os.listdir(dataPath):
		if patient.endswith('.mat'):
			trainingSet.append(patient)
			counter+=1
	print "%i samples found"%counter

	trainOut = [[1,0],[0,1]]*len(trainingSet) #this will contain the actual state of the brain

	data =[]
	for patient in trainingSet:
		temp = sio.loadmat('%s%s'%(dataPath,patient))
		data.append(temp['data']['X'][0][0][:1000])

	data = numpy.stack(data)
	trainOut = numpy.stack(trainOut)
	
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data

#be able to read from an Attentive folder and create their truth values
def getJsonData(dataPath):

	trainOut = numpy.array([[1,0]]) #this will contain the actual state of the brain
	data =[]
	res = {}
	with open(dataPath) as infile:
		res = json.load(infile)
	for timeStamp in res['data']:
		data.append(numpy.array(timeStamp['channel_values'],dtype='float32'))		
	data = numpy.stack(data,axis=1)
	data = numpy.resize(data,(data.shape[0],1000))
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer

	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)

	#These are the hidden layers mostly for demonstration purposes
	#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
	#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

	#network = lasagne.layers.Conv2DLayer(network, num_filters=2,pad='same', filter_size = (1,1), nonlinearity=softmax)

	return network

#---------------------------------For training------------------------------------------
def createTrainer(network,input_var,y):
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
	return train_function

def createValidator(network, input_var, y):
	#We will use this for validation
	testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
	testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

	validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage
	return validateFn

def saveModel(network,saveLocation='',modelName='brain1'):

	networkName = '%s%s.npz'%(saveLocation,modelName)
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network

def testNetwork(network,input_var,dataPath,trainingSet):
	out = lasagne.layers.get_output(network)
	test_fn = theano.function([input_var],out)

	data = getJsonData("%s%s"%(dataPath,trainingSet[2]))
	trainIn = data['input'].reshape([1,1] + list(data['input'].shape))

	print "Sample: ", trainingSet[2]
	print "Prediction: Attentive" if test_fn(trainIn)[0,0] == 1 else "Prediction: Not Attentive",
	

def main():
	dataPath = 'data/'
	#dataPath = '/home/rfratila/Desktop/MENTALdata/'
	input_var = T.tensor4('input')
	y = T.dmatrix('truth')
	trainFromScratch = True
	trainingSet = []
	counter=0

	for patient in os.listdir('%s%s'%(dataPath,'attentive')):
		import pudb; pu.db
		if patient.endswith('.json'):
			trainingSet.append(patient)
			counter+=1
	print "%i samples found"%counter

	data = getJsonData("%s%s"%(dataPath,trainingSet[0]))

	print ("Creating Network...")
	networkDimensions = (1,1,data['input'].shape[0],data['input'].shape[1])
	network  = createNetwork(networkDimensions, input_var)
	print ("Creating Trainer...")

	trainer = createTrainer(network,input_var,y)
	print ("Creating Validator...")
	validator = createValidator(network,input_var,y)

	if not trainFromScratch:
		print 'loading a previously trained model...\n'
		network = loadModel(network,'testing123.npz')

	iterations = 30
	print ("Training for %s iterations"%iterations)
	record = OrderedDict(iteration=[],error=[],accuracy=[])

	for i in xrange(iterations):
		chooseRandomly = numpy.random.randint(counter)
		print ("\nGathering data...%s"%trainingSet[chooseRandomly])
		data = getJsonData("%s%s"%(dataPath,trainingSet[chooseRandomly]))

		print "--> Iteration: %d"%(i)

		trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
		trainer(trainIn, data['truth'])
		
		error, accuracy = validator(trainIn, data['truth'])			     #pass modified data through network
		record['error'].append(error)
		record['accuracy'].append(accuracy)
		record['iteration'].append(i)
		print "	error: ",error,"and accuracy: ", accuracy

	testNetwork(network,input_var,dataPath,trainingSet)

	saveModel(network=network,modelName='testing123')
	pylab.plot(record['iteration'],record['error'], '-ro',label='TrErr')
	pylab.plot(record['iteration'],record['accuracy'],'-go',label='TrAcc')
	pylab.xlabel("Iteration")
	pylab.ylabel("Percentage")
	pylab.show(False)
main()