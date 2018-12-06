import xmlrpc.client
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE HYPERPARAMETERS TO PASS INTO TRAIN FUNCTION
HY = {
    # each element of conv_layers should be tuple:
    #(num filters, filtersize tuple, poolsize tuple OR none if not pooling)
    "conv_layers": [[(32,(3,3),(2, 2)), (10,(3,3),(2,2))], [(32,(3,3),(2, 2))]],
    # this is a list of sizes for each dense layer
    # do not include the final prediction layer here
    "dense_layers" : [[40,20],[10]],
    # apply the SAME activation and dropout to every conv and dense layer 
    # (except for final output which has hardcoded softmax activation w/ no dropout)
    "activation": ["relu", "sigmoid"], 
    "dropout":  [x * 0.1 for x in range(2)],
    # l2 regularization for weights
    "k_reg": [0,.001],
    # optimization hyperparameters
    "learning_rate": [.01],
    "grad_clip_norm": [x * 0.2 for x in range(2)]}

HY_small = {
    # each element of conv_layers should be tuple:
    #(num filters, filtersize tuple, poolsize tuple OR none if not pooling)
    "conv_layers": [[(32,(3,3),(2, 2))]],
    # this is a list of sizes for each dense layer
    # do not include the final prediction layer here
    "dense_layers" : [[10]],
    # apply the SAME activation and dropout to every conv and dense layer 
    # (except for final output which has hardcoded softmax activation w/ no dropout)
    "activation": ["sigmoid"], 
    "dropout":  [0.1],
    # l2 regularization for weights
    "k_reg": [0,.001],
    # optimization hyperparameters
    "learning_rate": [.01],
    "grad_clip_norm": [x * 0.2 for x in range(3)]}


with xmlrpc.client.ServerProxy('http://localhost:8800') as s:
    print(s)
    output = s.train_request(HY)
    np.savetxt("./final_loss_matrix_nofreeze.txt", np.asarray(output[0]), delimiter=",")
    print("The best validation loss was %f using hyperparameters %s and %i epochs. This model is saved at %s"%(output[2],str(output[3]),output[4],output[5]))
    for row in output[0]:
        plt.plot(range(1,len(row)+1), row)
    #plt.legend(np.arange(output[1]),title="Hyperparameter combination #",loc=1)
    plt.title('Losses from all models')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.show()
