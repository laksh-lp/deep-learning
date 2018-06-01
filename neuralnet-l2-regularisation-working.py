#NOTE that the python file should be run in the folder where image dataset is stored 

#NOTE that the python file should be run in the folder where image dataset is stored 

import cv2
import pickle
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
print(np.shape(train_set_x1))

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
print(np.shape(train_set_x2))


train_x = np.concatenate((train_set_x1,train_set_x2),1)
print(np.shape(train_x))

train_set_y1=np.ones((1,86))
train_set_y2=np.zeros((1,86))
train_y = np.concatenate((train_set_y1,train_set_y2),1)
print(np.shape(train_y))


first = cv2.imread('ball-test-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,15):
    s = 'ball-test-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
test_set_x1 = c
print(np.shape(test_set_x1))


first = cv2.imread('non-ball-test-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,15):
    s = 'non-ball-test-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
test_set_x2 = c
print(np.shape(test_set_x2))


test_x = np.concatenate((test_set_x1,test_set_x2),1)
print(np.shape(test_x))

test_set_y1=np.ones((1,15))
test_set_y2=np.zeros((1,15))
test_y = np.concatenate((test_set_y1,test_set_y2),1)
print(np.shape(test_y))

test_x = test_x/255.
train_x = train_x/255.


#################################################################################
# dataset has been created lets start further


#############################################################################starting functions are written in this block

def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*(2 / np.sqrt(layer_dims[l-1]))
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*(2/ np.sqrt(layer_dims[l-1]))
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


def initialize_parameters_deep(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) *(2/ np.sqrt(layer_dims[l-1])) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y , parameters):
    m = Y.shape[1]
    
    W1 = parameters['W'+str(1)]
    W2 = parameters['W'+str(2)]
    W3 = parameters['W'+str(3)]
    W4 = parameters['W'+str(4)]


    lambd = 0.7
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)) + (0.5*lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3))+np.sum(np.square(W4))))/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    lambd = 0.7
    dW = 1./m * np.dot(dZ,A_prev.T) + (lambd*W)/m
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


############################################################################end of initial functions


############################################################################the scene starts from here

layers_dims = [30000, 20, 7,5, 1]


def L_layer_model(X, Y, layers_dims, learning_rate = 0.005, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []                         # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y ,parameters)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


#################################################################################################lets predict now

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    probas, caches = L_model_forward(X, parameters)

    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


##################################################################################### WTF is this i dont know
def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))





############################################################################################











predictions = predict(train_x, train_y, parameters)

predictions2 = predict(test_x, test_y, parameters)




###########################################################################################################################
print ('Accuracy: %d' % float((np.dot(train_y,predictions.T) + np.dot(1-train_y,1-predictions.T))/float(train_y.size)*100) + '%')
print ('Accuracy: %d' % float((np.dot(test_y,predictions2.T) + np.dot(1-test_y,1-predictions2.T))/float(test_y.size)*100) + '%')



f = open('parameters.pckl', 'wb')
pickle.dump(parameters,f)
f.close()

