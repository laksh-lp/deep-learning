import cv2
import numpy as np

import matplotlib.pyplot as plt


first = cv2.imread('ball-train-set1.jpg')
c = np.reshape(first,(30000,1))
i = 2
for i in range(1,86):
    s = 'ball-train-set'+str(i)+'.jpg'
    image = cv2.imread(s,1)
    resh = np.reshape(image,(30000,1))
    c = np.concatenate((c,resh),1)
    previous=resh
train_set_x1 = c
print(np.shape(train_set_x1))


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


train_set_x = np.concatenate((train_set_x1,train_set_x2),1)
print(np.shape(train_set_x))

train_set_y1=np.ones((1,86))
train_set_y2=np.zeros((1,86))
train_set_y = np.concatenate((train_set_y1,train_set_y2),1)
print(np.shape(train_set_y))


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
print(np.shape(test_set_x1))


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
print(np.shape(test_set_x2))


test_set_x = np.concatenate((test_set_x1,test_set_x2),1)
print(np.shape(test_set_x))

test_set_y1=np.ones((1,22))
test_set_y2=np.zeros((1,22))
test_set_y = np.concatenate((test_set_y1,test_set_y2),1)
print(np.shape(test_set_y))

test_set_x = test_set_x/255.
train_set_x = train_set_x/255.

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)                                   # compute activation
    cost =(np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1-A).T))/(-m)                         # compute cost
    dw = np.dot(X,(A-Y).T)/m
    db = (np.sum(A - Y))/m
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        
        
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b) 
    for i in range(A.shape[1]):
        
        if A[0,i] <= 0.5:
             Y_prediction[0,i] = 0
        elif A[0,i] > 0.5:
             Y_prediction[0,i] = 1
        
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

