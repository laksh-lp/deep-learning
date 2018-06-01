#NOTE that the python file should be run in the folder where image dataset is stored 

import cv2
import numpy as np
import matplotlib.pyplot as plt

classes = ['non-ball','ball']
#ball-train-set1.jpg is the name of the image
first = cv2.imread('ball-train-set1.jpg')
#30000 = 100*100*3 = no. of rows* no. of column * dimensions
c = np.reshape(first,(30000,1))
i = 2
# 86 = m (no. of images in the dataset)
for i in range(1,86):
    s = 'ball-train-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
train_set_x1 = c
#print(np.shape(train_set_x1))

# all the loops below have the same concept and understanding as the first one 

first = cv2.imread('non-ball-train-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,86):
    s = 'non-ball-train-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
train_set_x2 = c
#print(np.shape(train_set_x2))


train_set_x = np.concatenate((train_set_x1,train_set_x2),1)
print("train_set_x = " + str(np.shape(train_set_x)))

train_set_y1=np.ones((1,86))
train_set_y2=np.zeros((1,86))
train_set_y = np.concatenate((train_set_y1,train_set_y2),1)
print("train_set_y = " + str(np.shape(train_set_y)))


first = cv2.imread('ball-test-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,22):
    s = 'ball-test-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
test_set_x1 = c
#print(np.shape(test_set_x1))


first = cv2.imread('non-ball-test-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,22):
    s = 'non-ball-test-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
test_set_x2 = c
#print(np.shape(test_set_x2))


test_set_x = np.concatenate((test_set_x1,test_set_x2),1)
print("test_set_x = " + str(np.shape(test_set_x)))

test_set_y1=np.ones((1,22))
test_set_y2=np.zeros((1,22))
test_set_y = np.concatenate((test_set_y1,test_set_y2),1)
print("test_set_y = " + str(np.shape(test_set_y)))

test_set_x = test_set_x/255.
train_set_x = train_set_x/255.



#################################################################################
# dataset has been created lets start further

X = train_set_x
Y = train_set_y


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

    
def layer_sizes(X, Y):
    n_x = np.shape(X)[0] 
    n_h = 4
    n_y = np.shape(Y)[0] 
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01 
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1] 
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = - np.sum(logprobs)/m
    
    cost = np.squeeze(cost)    
    assert(isinstance(cost, float))
    
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis = 1,keepdims = True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis = 1,keepdims = True)/m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
   
    for i in range(0, num_iterations):
         
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)


predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')








