import sys, os, time
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver, xmlrpc.client
#from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import ParameterGrid

print_graphs = False
max_epochs = 200
running_avg_window = 2
slopes_window = 4
min_observations = 3
tolerance_eps = 5

loss_quantile_cutoff = .5
slope_quantile_cutoff = .5


num_workers = 1


HY_COMBOS = {
	# each element of conv_layers should be tuple:
	#(num filters, filtersize tuple, poolsize tuple OR none if not pooling)
	"conv_layers": [[(32,(3,3),(2, 2)), (10,(3,3),(2,2))], [(32,(3,3),(2, 2))],[]],
	# this is a list of sizes for each dense layer
	# do not include the final prediction layer here
	"dense_layers" : [[50,10,5], [40,20],[10],[]],
	# apply the SAME activation and dropout to every conv and dense layer 
	# (except for final output which has hardcoded softmax activation w/ no dropout)
	"activation": ["relu", "sigmoid"], 
	"dropout":  np.arange(0,.5,.1),
	# l2 regularization for weights
	"k_reg": [0,.001],
	# optimization hyperparameters
	"learning_rate": [.01],
	"grad_clip_norm": np.arange(0,.5,.2)}
    
    
hy_list = list(ParameterGrid(HY_COMBOS))

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

	#def quit(self):
	#	server_connects = []
	#	for i in range(num_workers):
	#		server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))
	#	model_num = 0
	#	server_connects[worker_num[model_num]-8801].quit()
	#	server_connects[worker_num[model_num]-8801].quit()
	#	time.sleep(1)
	#	os.system('python worker.py '+str(worker_num[model_num])+ " &")
	#	return True
    
	def model_finished(self, model_num):
		WORKERS[np.where(WORKERS==model_num)] = np.nan
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
			server_connects[np.where(WORKERS==model_num)].quit()
			server_connects[np.where(WORKERS==model_num)].quit()
			time.sleep(1)
			os.system('python worker.py '+str(np.where(WORKERS==model_num)+8801)+ " &")
			time.sleep(1)
			WORKERS[selected_worker] = np.nan

		elif (ep_counters[model_num] >= max_epochs-1):
			print("MODEL %i DONE"%model_num)
			WORKERS[selected_worker] = np.nan
			    
		else:
			ep_counters[model_num] += 1
		return True

	def train_request(self, message):
		server_connects = []
		for i in range(num_workers):
			server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))

		# how often we want to generate a plot (total epochs run for any model)
		plotting_frequency = 50
		print_counter=0
		while True:

			if print_graphs and np.sum(list(ep_counters.values())) % plotting_frequency == 0 and np.sum(list(ep_counters.values())) !=0:
				if print_counter < 100:
					plt.figure()
					for row in losses:
						plt.plot(np.arange(200), row)
					plt.legend(np.arange(len(hy_list)))
					print_counter+=1

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
					#try:
					hy_list[new_mod]["dropout"]=float(hy_list[new_mod]["dropout"])
					hy_list[new_mod]["grad_clip_norm"]=float(hy_list[new_mod]["grad_clip_norm"])
					server_connects[new_worker_id].train(new_mod, hy_list[new_mod])
						#WORKERS[new_worker_id] = np.nan
					#except:
					#	print("worker error or stop")
						#time.sleep(5)
					#	continue

				else:
					print("here_million")
					break

		if print_graphs:
			for row in losses:
				plt.plot(np.arange(200), row)
			plt.legend(np.arange(len(hy_list)))

		return True




# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', 8800),logRequests=False) # Can process multiple requests at once
server.register_instance(MyFuncs())
server.serve_forever()


