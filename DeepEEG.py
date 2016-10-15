import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T

dataPath = '/home/rfratila/Desktop/MENTALdata/'
inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%dataPath)# got this from: http://bnci-horizon-2020.eu/database/data-sets
#inputMat = sio.loadmat('eeglab_data.set') #data is stored under inputMat['EEG']

trainOut = numpy.array([[1,0]]) #this will contain the actual state of the brain

input_var = T.tensor4('input')
y = T.dmatrix('truth')

data = numpy.array(inputMat['data']['X'][0][0][:10000], dtype='float32') #of size (657660, 62)
channels = 62
dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer

network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
#These are the hidden layers mostly for demonstration purposes
#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
#network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

#Trying to figure out how to extract the frequencies and time from the data
trainIn = data.reshape([1,1] + list(data.shape))
out = lasagne.layers.get_output(network)
view_fn = theano.function([input_var], out)
testInput = view_fn(trainIn)
print testInput
#import pudb; pu.db

#---------------------------------For training------------------------------------------

#get all parameters from network
params = lasagne.layers.get_all_params(network, trainable=True)

#calculate a loss function which has to be a scalar
cost = T.nnet.categorical_crossentropy(out, y).mean()

#calculate updates using ADAM optimization gradient descent
updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)

#theano function to compare brain to their masks with ADAM optimization
train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)

#We will use this for validation
testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage

for i in xrange(5):

	train_function(trainIn, trainOut)
	
	error, accuracy = validateFn(trainIn, trainOut) 			     #pass modified data through network

	print "error: ",error,"and accuracy: ", accuracy

testInput = view_fn(trainIn)
print testInput