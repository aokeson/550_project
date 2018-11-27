import sys, time, os, signal
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver
import xmlrpc.client
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import threading
from threading import Thread

#thread_id = 0

X = np.genfromtxt("../data/mnist.data", max_rows=100)
y = np.genfromtxt("../data/mnist.labels", max_rows=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("done loading data")

# reshape training data into 2d
X_train_c = X_train.reshape(len(X_train), 28, 28, 1)
X_test_c = X_test.reshape(len(X_test), 28, 28, 1)

leader = xmlrpc.client.ServerProxy('http://localhost:8800',allow_none=True)

class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

if len(sys.argv) == 2:
	server_number = int(sys.argv[1])
else:
	print("Must specify a server number")
	exit()

def training(model_num, HY):
	global thread_id
	thread_id = threading.get_ident()
	num_epochs = 5
        
	model = Sequential()

	# ADD CONVOLUTION LAYERS
	for i,c_params in enumerate(HY["conv_layers"]):
            
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
                    
	model.add(Flatten())

	for i,dense_nodes in enumerate(HY["dense_layers"]):

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
		time.sleep(5)

	history = model.fit(X_train_c, y_train, validation_data=(X_test_c, y_test), epochs=num_epochs, verbose=1, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=report_ep_loss)])


# Functions that client can ask server to do
class MyFuncs:
    
	def quit(self):
		global quit 
		quit = True
		#print(thread_id)
		#print(threading.get_ident())
		#signal.pthread_kill(thread_id, signal.SIGTERM)
		#os.kill(pid, signal.SIGUSR1)
		return "stopping"

	def train(self, model_num, HY):
		thread = Thread(target=training(model_num, HY))
		#thread_id = thread.getName()
		#print(threading.get_ident())
		#thread_id = thread.get_ident()
		thread.start()
		#time.sleep(1)

		#pid = os.getpid()
		#print(pid)

		
		# If we want early stopping, can add as callback above: 
		#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

		return "done"

# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', server_number),logRequests=False) # Can process multiple requests at once
#server = SimpleXMLRPCServer(('localhost', server_number)) # Puts recieved requests in a queue and then completes them serially
server.register_instance(MyFuncs())
#server.serve_forever()
quit = False


while not quit:
	server.handle_request()

print('server {} is dead'.format(server_number))
server.server_close()


import os
os.abort()
#print("after")
