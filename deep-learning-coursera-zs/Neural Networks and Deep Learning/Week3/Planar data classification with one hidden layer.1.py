# testCases provides some test examples to assess the correctness of your functions
# planar_utils provide various useful functions used in this assignment

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(2) # set a seed so that the results are consistent

# linear_model
def LogisticRegression():
    # - a numpy-array (matrix) X that contains your features (x1, x2)
    # - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
    X, Y = load_planar_dataset()

    # Visualize the data:
    # plt.scatter 画散列点图，参数c指定类别label，
    # cmap = plt.cm.Spectral实现的功能是给label为1的点一种颜色，给label为0的点另一种颜色。
    plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral)
    plt.show()

    ### START CODE HERE ### (≈ 3 lines of code)
    shape_X = X.shape
    shape_Y = Y.shape
    m = Y.shape[1]  # training set size
    ### END CODE HERE ###

    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))

    # Simple Logistic Regression
    # Train the logistic regression classifier
    # sklearn's built-in functions 
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)

    # Plot the decision boundary for logistic regression
    # plot_decision_boundary在planar_utils.py里
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
        '% ' + "(percentage of correctly labelled datapoints)")

# Neural Network model
# Reminder: The general methodology to build a Neural Network is to:

# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Implement forward propagation
#     - Compute loss
#     - Implement backward propagation to get the gradients
#     - Update parameters (gradient descent)

# 只有一个隐层的神经网络
# n为隐层节点数，dim为输入x的维度
def initializingParameters(n,dim):
    W1 = np.random.random((n,dim)) * 0.01
    B1 = np.zeros((n,1))
    W2 = np.random.random((1,n)) * 0.01
    B2 = np.zeros((1,1))

    return W1, B1, W2, B2

def tanh(x):
    a = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    return a

def d_tanh(x):
    return (1 - tanh(x) ** 2)

def think(X,W1,B1,W2,B2):
    Z1 = np.dot(W1,X) + B1
    A1 = tanh(Z1)

    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def lost_fun(A,Y):
    # A的值不能为0或1，log会取到负无穷
    # A[A<0.000000000001] = 0.000000000001
    # A[A>0.999999999999] = 0.999999999999
    lost = (-1/A.shape[1]) * np.sum((Y * np.log(A)) + ((1-Y) * np.log(1-A)))
    print("loss_fun: " + str(lost))
    return lost

def train(X,Y,W1,B1,W2,B2,loop_num,p=0.05):
    cost = []
    m = Y.shape[1]
    for i in range(loop_num):
        Z1, A1, Z2, A2 = think(X,W1,B1,W2,B2)

        if i % 100 == 0:
            print(str(i) + "times.")
            lost = lost_fun(A2,Y)
            cost.append(lost)
            
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2,A1.T)/m
        dB2 = np.sum(dZ2,axis = 1,keepdims=True)/m

        dZ1 = np.dot(W2.T,dZ2) * d_tanh(Z1)
        dW1 = np.dot(dZ1,X.T)/m
        dB1 = np.sum(dZ1,axis = 1 ,keepdims=True)/m

        W1 -= p * dW1
        B1 -= p * dB1
        W2 -= p * dW2
        B2 -= p * dB2
    
    return W1,B1,W2,B2

def predict(X,W1,B1,W2,B2):
    Z1, A1, Z2, A2 = think(X,W1,B1,W2,B2)
    predictions = np.round(A2)
    return predictions

def validat(X,Y,W1,B1,W2,B2):
    Z1, A1, Z2, A2 = think(X,W1,B1,W2,B2)
    A2 = np.round(A2)
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y, A2.T) + np.dot(1 - Y,1 - A2.T)) / float(Y.size) * 100) +
       '% ' + "(percentage of correctly labelled datapoints)")

if __name__ == '__main__':
    # 读取数据集
    X, Y = load_planar_dataset()

    W1,B1,W2,B2 = initializingParameters(4,X.shape[0])

    W1,B1,W2,B2 = train(X,Y,W1,B1,W2,B2,10000,1.2)

    validat(X,Y,W1,B1,W2,B2)
    
    plot_decision_boundary(lambda x: predict(x.T,W1,B1,W2,B2), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    pass