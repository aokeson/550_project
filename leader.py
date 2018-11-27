import sys
import os
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver
import xmlrpc.client
from multiprocessing import Process
from threading import Thread
import numpy as np
import time

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
		print("HERE")
		#time.sleep(5)
		print("after sleep")
		#thread = Thread(target=os.system('python worker.py '+str(worker_num[model_num])))
		#thread.start()
		os.system('python worker.py '+str(worker_num[model_num])+ " &")
		print("startup")
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
		server_connects[worker_num[this_model_num]-8801].train(this_model_num, message)
		finished[this_model_num] = True
		return True


# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', 8800),logRequests=False) # Can process multiple requests at once
#server = SimpleXMLRPCServer(('localhost', 8800)) # Puts recieved requests in a queue and then completes them serially
server.register_instance(MyFuncs())
server.serve_forever()


