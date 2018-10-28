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
    np.random.seed(1)
    W = [0] * (len(L))
    b = [0] * (len(L))
    # L是每层节点的个数，包括输入层
    for i in range(1,len(L)):
        # W[i] = np.random.random((L[i],L[i-1])) * 0.01
        # b[i] = np.random.random((L[i],1))
        W[i] = np.random.randn(L[i],L[i-1]) * 0.01
        b[i] = np.zeros((L[i], 1))
        
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

def Sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))

def think_relu(X,W,b):
    Z = np.dot(W,X) + b
    A = ReLU(Z)
    return Z,A

def think_sigmoid(X,W,b):
    Z = np.dot(W,X) + b
    A = Sigmoid(Z)
    return Z,A

def train(X,Y,W,b,L,p = 0.05,loop_num = 1500):
    Z = [0] * (len(L))
    dW = [0] * (len(L))
    db = [0] * (len(L))
    A = [0] * (len(L))
    A[0] = X
    lost = [0] * (len(L))
    for j in range(loop_num):
        for i in range(1,len(L)-1):
            Z[i],A[i] = think_relu(A[i-1],W[i],b[i])
        Z[len(L)-1],A[len(L)-1] = think_sigmoid(A[len(L)-2],W[len(L)-1],b[len(L)-1])


        if j % 100 == 0:
            lost.append(lost_fun(A[len(L)-1],Y))

        # dA = d_lost_fun(A,Y)
        # sigmoid 层反向计算
        dZ = A[len(L)-1] - Y   # = dA * d_sigmoid
        dW[len(L)-1] = np.dot(dZ,A[len(L)-2].T)/Y.shape[1]
        db[len(L)-1] = np.sum(dZ,axis =1 ,keepdims=True)/Y.shape[1]
        dA = np.dot(W[len(L)-1].T,dZ)
        # ReLU 层反向计算
        for i in range(len(L)-2,0,-1):
            dZ = dA * d_ReLU(Z[i])
            dW[i] = np.dot(dZ,A[i-1].T)/Y.shape[1]
            db[i] = np.sum(dZ,axis =1 ,keepdims=True)/Y.shape[1]
            dA = np.dot(W[i].T,dZ)

        for i in range(1,len(L)):
            W[i] -= p * dW[i]
            b[i] -= p * db[i]

    return W,b

def test(X,Y,W,b,L):
    Z = [0] * (len(L))
    A = X
    for i in range(1,len(L)-1):
            Z[i],A = think_relu(A,W[i],b[i])
    Z[len(L)-1],A = think_sigmoid(A,W[len(L)-1],b[len(L)-1])

    A = np.round(A)
    correct_num = np.sum((A-Y)==0)
    correct_precent = correct_num /Y.shape[1]
    return correct_num,correct_precent

def load_data():
    filepath = os.path.dirname(os.path.realpath(__file__)) + '\\'
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(filepath)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    test_set_x = test_set_x_flatten / 255
    return train_set_x, train_set_y, test_set_x, test_set_y 

if __name__ == "__main__":

    train_set_x, train_set_y, test_set_x, test_set_y = load_data()
    L = [train_set_x.shape[0],7,1]
    W,b = initializingParameters(L)

    W,b = train(train_set_x, train_set_y,W,b,L,0.0075,2500)

    correct_num,correct_precent = test(train_set_x, train_set_y,W,b,L)
    correct_num,correct_precent = test(test_set_x, test_set_y,W,b,L)

    print('correct_num:' + str(correct_num))
    print('correct_precent:' + str(correct_precent))

