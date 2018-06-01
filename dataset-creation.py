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

print(test_set_x)