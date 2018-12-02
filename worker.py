import sys, time, os, signal
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver, xmlrpc.client
import numpy as np
from threading import Thread
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout

leader = xmlrpc.client.ServerProxy('http://localhost:8800',allow_none=True)

class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

if len(sys.argv) == 2:
	server_number = int(sys.argv[1])
else:
	print("Must specify a server number")
	exit()

def training(model_num, HY):
	time.sleep(1)
	X = np.genfromtxt("../data/mnist.data", max_rows=100)
	y = np.genfromtxt("../data/mnist.labels", max_rows=100)
	#X = np.genfromtxt("../data/mnist.data")
	#y = np.genfromtxt("../data/mnist.labels")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	print("done loading data")

	# reshape training data into 2d
	X_train_c = X_train.reshape(len(X_train), 28, 28, 1)
	X_test_c = X_test.reshape(len(X_test), 28, 28, 1)
	
	num_epochs = 5
    
	model = Sequential()
	no_conv_layers = True
	print(HY)
	print(HY["conv_layers"])
	# ADD CONVOLUTION LAYERS
	for i,c_params in enumerate(HY["conv_layers"]):
		no_conv_layers = False

		num_filters, kernel_size, pooling_size = c_params

		# if it's the first layer, need to specify input shape
		if i==0: 
			model.add(Conv2D(num_filters, kernel_size, kernel_regularizer=keras.regularizers.l2(HY["k_reg"]), input_shape=(28,28,1)))
		else:
			model.add(Conv2D(num_filters, kernel_size, kernel_regularizer=keras.regularizers.l2(HY["k_reg"])))

		# add activation, pooling, dropout 
		model.add(Activation(HY["activation"]))
		if pooling_size:           
			model.add(MaxPooling2D(pool_size=pooling_size))
		if HY["dropout"]:
			model.add(Dropout(HY["dropout"]))

	if not no_conv_layers:
		model.add(Flatten())

	for i,dense_nodes in enumerate(HY["dense_layers"]):

		if no_conv_layers and i==0:
			print("HERE")
			model.add(Dense(dense_nodes, kernel_regularizer=keras.regularizers.l2(HY["k_reg"]), input_dim=784))
		else:
			model.add(Dense(dense_nodes, kernel_regularizer=keras.regularizers.l2(HY["k_reg"])))

		# add activation and dropout 
		model.add(Activation(HY["activation"]))
		if HY["dropout"]:
			model.add(Dropout(HY["dropout"]))


	# once all the hidden nodes are added, add the output layer with a softmax activation
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(optimizer=keras.optimizers.SGD(lr=HY["learning_rate"], clipnorm=HY["grad_clip_norm"]),  loss='categorical_crossentropy', metrics=['accuracy'])

	def report_ep_loss(ep,logs):
		print(ep, logs['val_loss'])
		leader.update(i, model_num, float(logs['val_loss']))
		#time.sleep(5)

	if no_conv_layers:
		model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, verbose=1, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=report_ep_loss)])
	else:
		model.fit(X_train_c, y_train, validation_data=(X_test_c, y_test), epochs=num_epochs, verbose=1, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=report_ep_loss)])
	leader.model_finished(model_num)


# Functions that client can ask server to do
class MyFuncs:
    
	def quit(self):
		global quit 
		quit = True
		return "stopping"

	def train(self, model_num, HY):
		Thread(target=training, args=(model_num,HY)).start()
		return "training"

# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', server_number),logRequests=False) # Can process multiple requests at once
server.register_instance(MyFuncs())

quit = False
while not quit:
	server.handle_request()
print('server {} is dead'.format(server_number))
server.server_close()
os.abort()
