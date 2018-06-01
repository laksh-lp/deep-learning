import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

f = open('store.pckl', 'rb')
w = pickle.load(f)
print(w)
f.close()

b = -0.0260428150792

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


img= cv2.imread('ball-train-set3.jpg')
rez = cv2.resize(img,(100,100))
c = np.reshape(rez,(30000,1))
x= c/255
A = sigmoid(np.dot(w.T,x) + b) 

if A <= 0.5:
	print('no match')

elif A > 0.5:
	print('match')


