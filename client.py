import xmlrpc.client
import sys
import random



s = xmlrpc.client.ServerProxy('http://localhost:8800')
print(s)

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

with xmlrpc.client.ServerProxy('http://localhost:8800') as s:
	if sys.argv[1] == "quit":
		print(s.quit())
	else:
		print(s.train_request(HY))