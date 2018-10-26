# numpy is the main package for scientific computing with Python.
# matplotlib is a library to plot graphs in Python.
# dnn_utils provides some necessary functions for this notebook.
# testCases provides some test cases to assess the correctness of your functions
# np.random.seed(1) is used to keep all the random function calls consistent. 
# It will help us grade your work. Please don't change the seed.
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import os
from lr_utils import load_dataset

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 1. Outline of the Assignment
# 2. Initialize the parameters for a two-layer network and for an $L$-layer neural network.
# 3. Implement the forward propagation module (shown in purple in the figure below).
#   Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
#   We give you the ACTIVATION function (relu/sigmoid).
#   Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#   Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
# 4. Compute the loss.
# 5. Implement the backward propagation module (denoted in red in the figure below).
#   Complete the LINEAR part of a layer's backward propagation step.
#   We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
#   Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
#   Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# 6. Finally update the parameters.

def initializingParameters(L):
    W = []
    b = []
    # L是每层节点的个数，包括输入层
    for i in range(1,L.size):
        W[i] = np.random.random((L[i],L[i-1])) * 0.001
        b[i] = np.random.random((L[i],1))
        
    return W,b

def lost_fun(A,Y):
    # A的值不能为0或1，log会取到负无穷
    # A[A<0.000000000001] = 0.000000000001
    # A[A>0.999999999999] = 0.999999999999
    lost = -(np.sum((Y * np.log(A)) + ((1-Y) * np.log(1-A)))/A.shape[1])
    print("loss_fun: " + str(lost))
    return lost
def d_lost_fun(A,Y):
    return (-Y/A + (1-Y)/(1-A))
    
def ReLU(Z):
    return np.maximum(0,Z)

def d_ReLU(Z):
    dZ = np.ones(Z.shape)
    dZ[Z<0] = 0
    return dZ


def think(X,W,b):
    Z = np.dot(W,X) + b
    A = relu(Z)
    return Z,A

def train(X,Y,W,b,L,p = 0.05,loop_num = 1500):
    Z = []
    dW = [] 
    db = []
    A = X
    lost = []
    for i in range(1500):
        for i in range(1,L.size+1):
            Z[i],A = think(A,W[i],b[i])

        if i % 100 == 0:
            lost.append(lost_fun(Z[L.size],Y))

        dA = d_lost_fun(A,Y)

        for i in range(L.size,0,-1):
            dZ = dA * d_ReLU(Z[i])
            dW[i] = np.dot(dZ,A[i-1].T)
            db[i] = np.sum(dZ,axis =1 ,keepdims=True)
            dA = np.dot(W[i].T,dZ)

        for i in range(1,L.size+1):
            W[i] -= p * dW[i]
            b[i] -= p * db[i]

    return W,b

def test(X,Y,W,b):
    Z = []
    A = X
    for i in range(1,W.shape[0]+1):
            Z[i],A = think(A,W[i],b[i])
    
    correct_num = np.sum((A-Y)==0)
    correct_precent = correct_num /Y.shape[0]
    return correct_num,correct_precent

def load_data():
    filepath = os.path.dirname(os.path.realpath(__file__)) + '\\'
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(filepath)
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y 