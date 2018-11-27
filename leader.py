import sys, os, time
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver, xmlrpc.client
#from threading import Thread
import numpy as np

num_workers = 1
next_model_num = 0

worker_num = [np.nan]
finished = [np.nan]
losses = [[np.nan,np.nan,np.nan,np.nan,np.nan]]

#server_connects = []
#for i in range(num_workers):
#	server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))

class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

def sendToTrain():
	return True

def keepRunning(model_num):
	# TODO after checkpoint
	return True
	#if np.random.rand() < 0.1:
	#	return False
	#else:
	#	return True


# Functions that client can ask server to do
class MyFuncs:

	def quit(self):
		server_connects = []
		for i in range(num_workers):
			server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))
		model_num = 0
		server_connects[worker_num[model_num]-8801].quit()
		time.sleep(1)
		os.system('python worker.py '+str(worker_num[model_num])+ " &")
		return True
    
	def update(self, epoch_num, model_num, loss):
		losses[0][epoch_num] = loss
		if keepRunning(model_num):
			return True
		else:
			server_connects[worker_num[model_num]].quit()
			os.system('python worker.py '+str(worker_num[model_num]))
			return True

	def train_request(self, message):
		server_connects = []
		for i in range(num_workers):
			server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))
		this_model_num = next_model_num
		#next_model_num += 1
		worker_num[this_model_num] = 8801
		finished[this_model_num] = False
		try:
			server_connects[worker_num[this_model_num]-8801].train(this_model_num, message)
		except:
			time.sleep(5)
			print("trying again")
			server_connects[worker_num[this_model_num]-8801].train(this_model_num, message)
		finished[this_model_num] = True
		return True


# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', 8800),logRequests=False) # Can process multiple requests at once
server.register_instance(MyFuncs())
server.serve_forever()


