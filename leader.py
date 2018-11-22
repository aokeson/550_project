import sys
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import socketserver
import xmlrpc.client
from multiprocessing import Process
from threading import Thread

import random

num_workers = 5

worker_num = []
finished = []
losses = []*10
models = []
server_connects = []
for i in range(num_workers):
	server_connects.append(xmlrpc.client.ServerProxy('http://localhost:'+str(8801+i),allow_none=True))

class MultiXMLRPCServer(socketserver.ThreadingMixIn,SimpleXMLRPCServer): pass

def sendToTrain():
    return

def keepRunning(model_num):
    # TODO
    return True


# Functions that client can ask server to do
class MyFuncs:
    
	def update(self, epoch_num, model_num, loss):
		losses[epoch_num][model_num] = loss
		if keepRunning(model_num):
			return True
		else:
			server_connects[worker_num[model_num]].quit()
			return True

	def train_request(self, message):
		
		return True


# Start server, register functions, serve continuously
server = MultiXMLRPCServer(('localhost', 8800),logRequests=False) # Can process multiple requests at once
#server = SimpleXMLRPCServer(('localhost', 8800)) # Puts recieved requests in a queue and then completes them serially
server.register_instance(MyFuncs())
server.serve_forever()


