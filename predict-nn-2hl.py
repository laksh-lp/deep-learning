import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


classes = ['non-ball','ball']


f = open('parameters.pckl', 'rb')
w = pickle.load(f)
print(w)
f.close()

