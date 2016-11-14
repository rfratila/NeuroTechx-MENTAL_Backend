import json
import numpy
import pylab #for graphing

data = {}
modelName = "testtrain"
with open('%sstats.json'%modelName, 'r') as infile:
		data = json.load(infile)

for key in data.keys():
	data[key] = numpy.array(data[key],dtype='float32')

pylab.plot(data['iteration'],data['error'], '-ro',label='TrErr')
pylab.plot(data['iteration'],data['accuracy'],'-go',label='TrAcc')
pylab.xlabel("Iteration")
pylab.ylabel("Percentage")
pylab.ylim(0,2)
savefig('.png'%modelName)
pylab.show()#enter param False if running in iterative mode