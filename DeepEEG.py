import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T

#inputMat = sio.loadmat('covertShiftsOfAttention_VPiac.mat')
inputMat = sio.loadmat('eeglab_data.set') #data is stored under inputMat['EEG']

import pudb; pu.db
input_var = T.tensor4('input')
y = T.dmatrix('truth')

network = lasagne.layers.InputLayer(shape=(None, 1, None, None),input_var=input_var)
#These are the hidden layers mostly for demonstration purposes
network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
#network = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

#Trying to figure out how to extract the frequencies and time from the data
test = numpy.asarray([1,1] + list(inputMat['EEG']))
out = lasagne.layers.get_output(network)
view_fn = theano.function([input_var], out)

#---------------------------------For training------------------------------------------

#get all parameters from network
params = lasagne.layers.get_all_params(network, trainable=True)

#calculate a loss function which has to be a scalar
cost = T.nnet.categorical_crossentropy(out, y).mean()

#calculate updates using ADAM optimization gradient descent
updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)

#theano function to compare brain to their masks with ADAM optimization
train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
val_fn = theano.function([input_var, y], [test_loss, test_acc])	 #check for error and accuracy percentage


test_prediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,y).mean()   #check how much error in prediction
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

train_function(trainIn, trainOut)
	
error, accuracy = val_fn(trainIn, trainOut) 			     #pass modified img through network

print "error: ",error,"and accuracy: ", accuracy