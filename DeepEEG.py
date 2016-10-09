import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T

#inputMat = sio.loadmat('covertShiftsOfAttention_VPiac.mat')
inputMat = sio.loadmat('eeglab_data.set') #data is stored under inputMat['EEG']


input_var = T.tensor4('input')
y = T.dmatrix('truth')

network = lasagne.layers.InputLayer(shape=(None, 1, 1, 1),input_var=input_var)
#These are the hidden layers mostly for demonstration purposes
network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify)
#network = lasagne.layers.Conv2DLayer(network, num_filters=2, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

#Trying to figure out how to extract the frequencies and time from the data
test = numpy.asarray([1,1] + list(inputMat['EEG']))
out = lasagne.layers.get_output(network)
view_fn = theano.function([input_var], out)