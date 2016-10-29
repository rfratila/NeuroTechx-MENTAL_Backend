import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os

from collections import OrderedDict

#TODO:
#Save some recordings for validation
#Be able to save model and load it back

#inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%dataPath)# got this from: http://bnci-horizon-2020.eu/database/data-sets
def getData(dataPath):

	trainingSet =[]
	counter = 0
	for patient in os.listdir(dataPath):
		if patient.endswith('.mat'):
			trainingSet.append(patient)
			counter+=1
	print "%i samples found"%counter

	trainOut = numpy.array([[1,0]]*len(trainingSet), dtype='float32') #this will contain the actual state of the brain

	data =[]
	for patient in trainingSet:
		temp = sio.loadmat('%s%s'%(dataPath,patient))

		data.append(numpy.array(temp['data']['X'][0][0][:10000], dtype='float32'))

	data = numpy.stack(data)
	data = OrderedDict(input=numpy.array(data), truth=numpy.array(trainOut))
	return data

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	#These are the hidden layers mostly for demonstration purposes
	#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
	#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	return network


#view_fn = theano.function([input_var], out)


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
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(self.network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network

def main():
	dataPath = '/home/rfratila/Desktop/MENTALdata/'
	input_var = T.tensor4('input')
	y = T.dmatrix('truth')

	print ("Gathering data..."),
	data = getData(dataPath)
	print ("Creating Network...")
	network  = createNetwork((1,1,data['input'].shape[1],data['input'].shape[2]), input_var)
	print ("Creating Trainer...")
	trainer = createTrainer(network,input_var,y)
	print ("Creating Validator...")
	validator = createValidator(network,input_var,y)

	iterations = 10
	print ("Training for %s iterations"%iterations)
	for i in xrange(iterations):
		chooseRandomly = numpy.random.randint(data['input'].shape[0])
		trainIn = data['input'][chooseRandomly].reshape([1,1] + list(data['input'][chooseRandomly].shape))
		#import pudb; pu.db
		trainer(trainIn, numpy.expand_dims(data['truth'][chooseRandomly], axis=0))
		
		error, accuracy = validator(trainIn, numpy.expand_dims(data['truth'][chooseRandomly], axis=0)) 			     #pass modified data through network

		print "error: ",error,"and accuracy: ", accuracy
main()