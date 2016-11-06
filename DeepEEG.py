import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os

from collections import OrderedDict
import pylab #for graphing

import json
from random import shuffle

#TODO:
#have a fully convolutional layer to account for variable recording time
#save some samples for testing and validation

# Custom softmax function to support fully convolutional networks
def softmax(x):
	e_x = theano.tensor.exp(x - x.max(axis=1, keepdims=True))
	return e_x / e_x.sum(axis=1, keepdims=True)

class PermuteLayer(lasagne.layers.Layer):
	def get_output_for(self, input, **kwargs):
		inShape = input.shape
		temp = input                        #(batch, filter, y, x)
		temp = temp.dimshuffle(1, 0, 2, 3)  #(filter, batch, y, x)
		temp = temp.flatten(2)              #(filter, batch)
		temp = temp.dimshuffle(1, 0)        #(batch, filter)
		return temp

	def get_output_shape_for(self, input_shape):
		return (input_shape[0] * numpy.product(input_shape[2:]) ,input_shape[1])


#inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%dataPath)# got this from: http://bnci-horizon-2020.eu/database/data-sets
def getData(dataPath):

	trainingSet =[]

	for patient in os.listdir(dataPath):
		if patient.endswith('.mat'):
			trainingSet.append(patient)

	print "%i samples found"%len(trainingSet)

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
	if 'inattentive' in dataPath:
		trainOut = numpy.array([[0,1]]) #this will contain the actual state of the brain: inattentive
	else:
		trainOut = numpy.array([[1,0]]) #this will contain the actual state of the brain: attentive
	data =[]
	res = {}
	with open(dataPath) as infile:
		res = json.load(infile)
	for timeStamp in res['data']:
		data.append(numpy.array(timeStamp['channel_values'],dtype='float32'))		
	data = numpy.stack(data,axis=1)
	data = numpy.resize(data,(data.shape[0],1500))
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	dense=True
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	print 'Input Layer:'
	print '	',lasagne.layers.get_output_shape(network)
	print 'Hidden Layer:'
	if dense:
		network = lasagne.layers.DenseLayer(network, num_units=800, nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.DropoutLayer(network,p=0.2)
		print '	',lasagne.layers.get_output_shape(network)

		network = lasagne.layers.DenseLayer(network, num_units=800, nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.DropoutLayer(network,p=0.2)
		print '	',lasagne.layers.get_output_shape(network)
	else:
		network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
		print '	',lasagne.layers.get_output_shape(network)

		network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
		print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	#network = lasagne.layers.Conv2DLayer(network, num_filters=2,pad='same', filter_size = (1,1), nonlinearity=softmax)
	#network = PermuteLayer(network)
	print 'Output Layer:'
	print '	',lasagne.layers.get_output_shape(network)
	#import pudb; pu.db

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
	print 'Saving model as',networkName
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network

def validateNetwork(network,input_var,validationSet):
	print '\n Validating the network'
	out = lasagne.layers.get_output(network)
	test_fn = theano.function([input_var],out)
	for sample in validationSet:
		data = getJsonData(sample)
		trainIn = data['input'].reshape([1,1] + list(data['input'].shape))

		print "Sample: ", sample
		print "Prediction: Attentive" if test_fn(trainIn)[0,0] == 1 else "Prediction: Inattentive"

def main():
	dataPath = 'data'
	#dataPath = '/home/rfratila/Desktop/MENTALdata/'
	testReserve = 0.2
	validationReserve = 0.2
	trainingReserve = 1-(testReserve+validationReserve)
	input_var = T.tensor4('input')
	y = T.dmatrix('truth')
	trainFromScratch = True
	dataSet = []

	for patient in [dataPath]:

		attentivePath = os.path.join(dataPath,'attentive')
		inattentivePath = os.path.join(dataPath,'inattentive')

		if os.path.exists(attentivePath) and os.path.exists(inattentivePath):
			dataSet += [os.path.join(attentivePath,i) for i in os.listdir(attentivePath)]
			dataSet += [os.path.join(inattentivePath,i) for i in os.listdir(inattentivePath)  if i.endswith('.json')]
			shuffle(dataSet)

	print "%i samples found"%len(dataSet)
	#This reserves the correct amount of samples for training, testing and validating
	trainingSet = dataSet[:int(trainingReserve*len(dataSet))]
	testSet = dataSet[int(trainingReserve*len(dataSet)):-int(testReserve*len(dataSet))]
	validationSet = dataSet[int(testReserve*len(dataSet) + int(trainingReserve*len(dataSet))):]

	inputDim = getJsonData(trainingSet[0])

	print ("Creating Network...")
	networkDimensions = (1,1,inputDim['input'].shape[0],inputDim['input'].shape[1])
	network  = createNetwork(networkDimensions, input_var)

	print ("Creating Trainer...")
	trainer = createTrainer(network,input_var,y)

	print ("Creating Validator...")
	validator = createValidator(network,input_var,y)

	if not trainFromScratch:
		print 'loading a previously trained model...\n'
		network = loadModel(network,'testing123.npz')

	epochs = 10
	samplesperEpoch = 10

	print ("Training for %s epochs with %s samples per epoch"%(epochs,samplesperEpoch))
	record = OrderedDict(iteration=[],error=[],accuracy=[])

	for epoch in xrange(epochs):
		print "--> Epoch: %d"%(epoch)
		for i in xrange(samplesperEpoch):
			chooseRandomly = numpy.random.randint(len(trainingSet))
			data = getJsonData(trainingSet[chooseRandomly])
			trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
			trainer(trainIn, data['truth'])

		chooseRandomly = numpy.random.randint(len(testSet))
		print ("Gathering data...%s"%testSet[chooseRandomly])
		data = getJsonData(testSet[chooseRandomly])
		trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
		error, accuracy = validator(trainIn, data['truth'])			     #pass modified data through network
		record['error'].append(error)
		record['accuracy'].append(accuracy)
		record['iteration'].append(epoch)
		print "	error: ",error,"and accuracy: ", accuracy

	validateNetwork(network,input_var,validationSet)

	saveModel(network=network,modelName='testing123')
	pylab.plot(record['iteration'],record['error'], '-ro',label='TrErr')
	pylab.plot(record['iteration'],record['accuracy'],'-go',label='TrAcc')
	pylab.xlabel("Iteration")
	pylab.ylabel("Percentage")
	pylab.ylim(0,2)
	pylab.show()#enter param False if running in iterative mode

main()