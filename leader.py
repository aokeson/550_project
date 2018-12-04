from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver, xmlrpc.client
import numpy as np
from sklearn.model_selection import ParameterGrid
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
import math, os

max_epochs = 200
running_avg_window = 2
slopes_window = 4
min_observations = 3
tolerance_eps = 5

loss_quantile_cutoff = .5
slope_quantile_cutoff = .5


num_workers = 3



class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

def check_stopping(model_num):
	# get the epochs for the current model that is not nan (i.e., avoid errors if we have dropped messages)
	mod_non_nan_loss_eps = np.where(~np.isnan(losses[model_num]))[0]

	# if there are not enough epochs run to make a decision, skip all of this and return False
	if len(mod_non_nan_loss_eps) >= tolerance_eps:

		# select chunk of loss matrix that corresponds to the most recent (#tolerance eps)
		# epochs that our current model has data for
		all_losses = losses[:, mod_non_nan_loss_eps[-tolerance_eps:]]

		# get all the slopes at the most recent epoch we have the current model's data for
		all_current_slopes = running_slopes[:, mod_non_nan_loss_eps[-1]]

		# check if we have enough runs from other models to make a decision. If we dont, just keep going:
		if np.min(np.sum(~np.isnan(all_losses),axis=0)) < min_observations:
		    return False

		else:
			# get quantiles for the current window 
			loss_quantiles = np.nanquantile(all_losses, loss_quantile_cutoff, axis=0)  #this is an array of length tolerance_eps
			slope_quantiles = np.nanquantile(all_current_slopes, slope_quantile_cutoff)  #this is a single value

			# if our loss is above quantile for all epochs we checked AND 
			# the slope is not as steep (not as negative) as the quantiles, stop
			#print(loss_quantiles, all_losses[model_num], slope_quantiles, all_current_slopes[model_num])
			if np.mean(all_losses[model_num] >= loss_quantiles) == 1  and  all_current_slopes[model_num] >= slope_quantiles:
				return True
			else:
				return False
	else:
		return False

# Functions that client can ask server to do
class MyFuncs:
    
	def model_finished(self, model_num):
		print("MODEL %i DONE"%model_num)
		WORKERS[np.where(WORKERS==model_num)[0][0]] = np.nan
		return True

	def update(self, epoch_num, model_num, loss):
		# update loss value
		losses[model_num, epoch_num] = loss

		# keep running average for the slopes --> then update the slope
		# if statements prevent us from indexing into negative indices of the array 
		if epoch_num - running_avg_window + 1 >= 0:
			running_averages[model_num, epoch_num] = np.mean(losses[model_num, epoch_num-running_avg_window+1:epoch_num+1])
		if epoch_num - slopes_window >= 0:
			running_slopes[model_num, epoch_num] = running_averages[model_num, epoch_num] - running_averages[model_num, epoch_num-slopes_window]

		if check_stopping(model_num):
			print("QUITTING MODEL %i"%model_num)
			server_connects = []
			for i in range(num_workers):
				server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))
			server_connects[np.where(WORKERS==model_num)[0][0]].quit()
			WORKERS[np.where(WORKERS==model_num)[0][0]] = np.nan
			    
		else:
			ep_counters[model_num] += 1
		return True

	def train_request(self, message):
		global hy_list
		global losses
		global running_averages
		global running_slopes
		global MODEL_QUEUE
		global WORKERS
		global ep_counters
		hy_list = list(ParameterGrid(message))

		# INITIALIZATION 
		losses = np.empty([len(hy_list), max_epochs])
		losses.fill(np.nan)

		running_averages = np.empty([len(hy_list), max_epochs])
		running_averages.fill(np.nan)

		running_slopes = np.empty([len(hy_list), max_epochs])
		running_slopes.fill(np.nan)

		# here, we're just pre-specifying a random order in which to test all the hyperparameters 
		MODEL_QUEUE = list(np.random.permutation(len(hy_list)))


		# array of length num_workers 
		# --> if the worker is idle, value is np.nan; otherwise, the value is the model number that's being trained
		WORKERS = np.empty(num_workers)
		WORKERS.fill(np.nan)


		# For the simulation, We randomly pick a worker to "receive" a message from, but in reality we just select data from 
		# the pre-determined training curve and pass it to the update function 
		# ep_counters is used to keep track of what epoch we're up to in training  
		# (it's a dict where each key is a model number and the value is the #epoch training is thought to be on)
		ep_counters = {}
		for mod in range(len(hy_list)):
			ep_counters[mod] = 0


		server_connects = []
		for i in range(num_workers):
			server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))

		# how often we want to generate a plot (total epochs run for any model)
		plotting_frequency = 50
		print_counter=0
		while True:

			# if there are no workers working on any models AND there are no models left to test, break out of the loop
			if len(np.where(~np.isnan(WORKERS))[0]) == 0 and not MODEL_QUEUE:
				break

			# if there are any unoccupied workers, add a new model from the queue
			while len(np.where(np.isnan(WORKERS))[0]) > 0:
				# make sure model queue isnt empty
				if MODEL_QUEUE:
					new_mod = int(MODEL_QUEUE.pop())
					new_worker_id = int(np.random.choice(np.where(np.isnan(WORKERS))[0]))
					print("starting new model %i on worker %i"%(new_mod, new_worker_id))
					WORKERS[new_worker_id] = new_mod
					hy_list[new_mod]["dropout"]=float(hy_list[new_mod]["dropout"])
					hy_list[new_mod]["grad_clip_norm"]=float(hy_list[new_mod]["grad_clip_norm"])
					server_connects[new_worker_id].train(new_mod, hy_list[new_mod], max_epochs)

				else:
					break


		HY = hy_list[math.floor(np.nanargmin(losses)/max_epochs)]

		X_train = np.genfromtxt("../data/mnist.data.train")#, max_rows=80)
		y_train = np.genfromtxt("../data/mnist.labels.train")#, max_rows=80)
		X_test = np.genfromtxt("../data/mnist.data.test")#, max_rows=20)
		y_test = np.genfromtxt("../data/mnist.labels.test")#, max_rows=20)
		X_train_c = X_train.reshape(len(X_train), 28, 28, 1)
		X_test_c = X_test.reshape(len(X_test), 28, 28, 1)
		
		num_epochs = int((np.nanargmin(losses)%max_epochs)+1)

		model = Sequential()

		no_conv_layers = True
		no_dense_layers = True
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
			no_dense_layers = False

			if no_conv_layers and i==0:
				model.add(Dense(dense_nodes, kernel_regularizer=keras.regularizers.l2(HY["k_reg"]), input_dim=784))
			else:
				model.add(Dense(dense_nodes, kernel_regularizer=keras.regularizers.l2(HY["k_reg"])))

			# add activation and dropout 
			model.add(Activation(HY["activation"]))
			if HY["dropout"]:
				model.add(Dropout(HY["dropout"]))

		# once all the hidden nodes are added, add the output layer with a softmax activation
		if no_conv_layers and no_dense_layers:
			model.add(Dense(10, input_dim=784))
		else:
			model.add(Dense(10))
		model.add(Activation('softmax'))

		model.compile(optimizer=keras.optimizers.SGD(lr=HY["learning_rate"], clipnorm=HY["grad_clip_norm"]),  loss='categorical_crossentropy', metrics=['accuracy'])

		save_path = str(os.path.dirname(os.path.abspath(__file__)))+str("/best_model.hdf5")
		if no_conv_layers:
			model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, verbose=1, callbacks=[keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=num_epochs)])
		else:
			model.fit(X_train_c, y_train, validation_data=(X_test_c, y_test), epochs=num_epochs, verbose=1, callbacks=[keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=num_epochs)])
		keras.backend.clear_session()

		return (losses.tolist(), len(hy_list), float(losses.tolist()[math.floor(np.nanargmin(losses)/max_epochs)][num_epochs-1]), HY, num_epochs, save_path)




# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', 8800),logRequests=False) # Can process multiple requests at once
server.register_instance(MyFuncs())
server.serve_forever()


