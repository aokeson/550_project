import xmlrpc.client


#s = xmlrpc.client.ServerProxy('http://localhost:8800')
#print(s)

# EXAMPLE HYPERPARAMETERS TO PASS INTO TRAIN FUNCTION
HY = {
    # each element of conv_layers should be tuple:
    #(num filters, filtersize tuple, poolsize tuple OR none if not pooling)
    "conv_layers": [[(32,(3,3),(2, 2)), (10,(3,3),(2,2))], [(32,(3,3),(2, 2))],[]],
    # this is a list of sizes for each dense layer
    # do not include the final prediction layer here
    "dense_layers" : [[50,10,5], [40,20],[10],[]],
    # apply the SAME activation and dropout to every conv and dense layer 
    # (except for final output which has hardcoded softmax activation w/ no dropout)
    "activation": ["relu", "sigmoid"], 
    "dropout":  [x * 0.1 for x in range(5)],
    # l2 regularization for weights
    "k_reg": [0,.001],
    # optimization hyperparameters
    "learning_rate": [.01],
    "grad_clip_norm": [x * 0.2 for x in range(3)]}


with xmlrpc.client.ServerProxy('http://localhost:8800') as s:
    print(s)
    print(s.train_request(HY))