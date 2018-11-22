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


# EXAMPLE HYPERPARAMETERS TO PASS INTO TRAIN FUNCTION
HY = {
     # each element of conv_layers should be tuple:
     #(num filters, filtersize tuple, poolsize tuple OR none if not pooling)
    "conv_layers": [(32,(3,3),(2, 2)), (10,(3,3),(2,2))],
    # this is a list of sizes for each dense layer
    # do not include the final prediction layer here
    "dense_layers" : [5,3],
    # apply the SAME activation and dropout to every conv and dense layer 
    # (except for final output which has hardcoded softmax activation w/ no dropout)
    "activation": "relu", 
    "dropout":  .1,
    # l2 regularization for weights
    "k_reg": .00001,
    # optimization hyperparameters
    "learning_rate": .01,
    "grad_clip_norm": .5}


X = np.genfromtxt("../data/mnist.data")
y = np.genfromtxt("../data/mnist.labels")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

# Functions that client can ask server to do
class MyFuncs:
    
    def quit(self):
        global quit 
        quit = True
        return "stopping"

    def train(self, model_num, HY):
        num_epochs = 10
        
        model = Sequential()

        # ADD CONVOLUTION LAYERS
        for i,c_params in enumerate(HY["conv_layers"]):
            
            num_filters, kernel_size, pooling_size = c_params
            
            # if it's the first layer, need to specify input shape
            if i==0:
                model.add(Conv2D(num_filters, kernel_size, kernel_regularizer=regularizers.l2(HY["k_reg"]), input_shape=(28,28,1)))
            else:
                model.add(Conv2D(num_filters, kernel_size, kernel_regularizer=regularizers.l2(HY["k_reg"])))
            
            # add activation, pooling, dropout 
            model.add(Activation(HY["activation"]))
            if pooling_size:           
                model.add(MaxPooling2D(pool_size=pooling_size))
            if HY["dropout"]:
                model.add(Dropout(HY["dropout"]))
                    
        model.add(Flatten())

        for i,dense_nodes in enumerate(HY["dense_layers"]):

            model.add(Dense(dense_nodes, kernel_regularizer=regularizers.l2(HY["k_reg"])))
            
            # add activation and dropout 
            model.add(Activation(HY["activation"]))
            if HY["dropout"]:
                model.add(Dropout(HY["dropout"]))


        # once all the hidden nodes are added, add the output layer with a softmax activation
        model.add(Dense(10))
        model.add(Activation('softmax'))


        model.compile(optimizer=optimizers.SGD(lr=HY["learning_rate"], clipnorm=HY["grad_clip_norm"]),  
                      loss='categorical_crossentropy', metrics=['accuracy'])

        def report_ep_loss(ep,logs):
            print(ep, logs['val_loss'])
            leader.update(i, model_num, logs['val_loss'])

        history = model.fit(X_train_c, y_train, validation_data=(X_test_c, y_test), epochs=num_epochs, verbose=1, 
                            callbacks=[LambdaCallback(on_epoch_end=report_ep_loss)])

        # If we want early stopping, can add as callback above: 
        #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        
        return [model_num, model]
        
        

# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', server_number),logRequests=False) # Can process multiple requests at once
#server = SimpleXMLRPCServer(('localhost', server_number)) # Puts recieved requests in a queue and then completes them serially
server.register_instance(MyFuncs())
server.serve_forever()
