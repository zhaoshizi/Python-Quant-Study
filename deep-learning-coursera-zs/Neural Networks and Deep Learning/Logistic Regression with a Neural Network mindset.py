import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import os

# I.Common steps for pre-processing a new dataset are:

# 1. Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# 2. Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
# 3. "Standardize" the data

# Loading the data (cat/non-cat)
filepath = os.path.dirname(os.path.realpath(__file__)) + '\\'
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(filepath)

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

# - m_train (number of training examples)
# - m_test (number of test examples)
# - num_px (= height = width of a training image)
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use:
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# One common preprocessing step in machine learning is to center and standardize your dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# II. General Architecture of the learning algorithm
# Key steps: In this exercise, you will carry out the following steps:
# 1. Initialize the parameters of the model
# 2. Learn the parameters for the model by minimizing the cost  
# 3. Use the learned parameters to make predictions (on the test set)
# 4. Analyse the results and conclude

# The main steps for building a Neural Network are:

# Define the model structure (such as number of input features)
# Initialize the model's parameters
# Loop:
#   Calculate current loss (forward propagation)
#   Calculate current gradient (backward propagation)
#   Update parameters (gradient descent)

def sigmod(z):
    # 应用sigmod激活函数
    return 1 / (1 + np.exp(-z))

def initializingParameters(dim):
    W = np.random.random((dim,1)) * 0.01
    #B = np.random.random((1,1)) * 0.01
    B = 0

    assert(W.shape == (dim, 1))
    assert(isinstance(B, float) or isinstance(B, int))
    return W,B

def loss_fun(A,Y):
    loss = np.sum(np.dot(Y,np.log(A)) + np.dot((1-Y),np.log(1-A)))/A.shape[1]
    print("loss_fun: " + str(loss))

def train(train_set_x,train_set_y,W,B):

    # 学习率
    i = 0.5
    for j in range(15000):
        A,Z = think(train_set_x,W,B)

        if i % 1000 == 0:
            loss_fun(A,train_set_y)

        dZ = A - train_set_y

        dw = np.dot(train_set_x,dZ.T)/m_train

        db = np.sum(dZ)/m_train

        W = W - i*dw

        B = B - i*db

    return W,B



def think(train_set_x,W,B):
    Z = np.dot(W.T,train_set_x)

    A = sigmod(Z)

    return A,Z

def test(test_set_x,test_set_y,W,B):
    
    test_A,test_Z = think(test_set_x,W,B)

    result = test_A ^ test_set_y
    succeed_cnt = np.sum(result)

    succeed_percent = succeed_cnt / test_set_y.shape[1]

    print("succeed: %d, succeed percent: %f" % succeed_cnt,succeed_percent)
