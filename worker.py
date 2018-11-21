import sys
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver
import xmlrpc.client
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

X = np.genfromtxt("mnist.data")
y = np.genfromtxt("mnist.labels")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

leader = xmlrpc.client.ServerProxy('http://localhost:8800',allow_none=True)

class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

if len(sys.argv) == 2:
	server_number = int(sys.argv[1])
else:
	print("Must specify a server number")
	exit()

# Functions that client can ask server to do
class MyFuncs:
    
	def quit(self):
		global quit 
		quit = True
		return "stopping"

	def train(self, model_num, model_params):
		num_epochs = 10
		trainX2d = 0
		
		model = Sequential()
		for i in range(len(model_params)):
			if model_params[i][0] == "Dense":
				if i == 0:
					model.add(Dense(model_params[i][1], activation=model_aprams[i][2], input_dim=784))
				else:
					model.add(Dense(model_params[i][1], activation=model_aprams[i][2]))
			elif model_params[i][0] == "Conv2D":
				if i == 0:
					trainX2d = trainX.reshape(len(X_train),28,28,1)
					testX2d = testX.reshape(len(X_test),28,28,1)
					model.add(Conv2D(filters=model_params[i][1], kernel_size=(model_params[i][2],model_params[i][2]), strides=(model_params[i][3],model_params[i][3]), input_shape=(28,28,1)))
				else:
					model.add(Conv2D(filters=model_params[i][1], kernel_size=(model_params[i][2],model_params[i][2]), strides=(model_params[i][3],model_params[i][3])))
			elif model_params[i][0] == "Flatten":
				model.add(Flatten())
			elif model_params[i][0] == "compile":
				model.compile(optimizer=model_params[i][1], loss=model_params[i][2], metrics=['accuracy'])
			else:
				return model_params[i][0]+" is not a defined Keras input"
		
		if trainX2d == 0:
			for i in range(num_epochs):
				history = model.fit(trainX, trainY, epochs=1, verbose=2, validation_data=(testX, testY))
				leader.update(i, model_num, history)
		else:
			for i in range(num_epochs):
				history = model.fit(trainX2d, trainY, epochs=1, verbose=2, validation_data=(testX2d, testY))
				leader.update(i, model_num, history)
		
		return [model_num, model]

# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', server_number),logRequests=False) # Can process multiple requests at once
#server = SimpleXMLRPCServer(('localhost', server_number)) # Puts recieved requests in a queue and then completes them serially
server.register_instance(MyFuncs())
server.serve_forever()