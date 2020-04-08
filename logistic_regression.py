import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
import pandas as pd

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def calc_cov_mat(ip_mat):
    mean_mat = np.sum(ip_mat, axis=0)/ip_mat.shape[0]
    mat_b = ip_mat - mean_mat
    return np.dot(mat_b.transpose(), mat_b)/(ip_mat.shape[0] - 1)


x_train=[]
y_train=[]
train_file_loc = sys.argv[1]
test_file_loc = sys.argv[2]

f = open(train_file_loc, 'r')
for ele in f.readlines():
    path, label = ele.strip().split()
    gray = mpimg.imread(path)
    gray = rgb2gray(gray)
    gray = cv2.resize(gray, (80,80), interpolation = cv2.INTER_AREA)
    x_train.append(gray.flatten())
    y_train.append(label)
x_train = np.array(x_train)
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
y_train = np.array(y_train)
f.close()


x_valid=[]
f = open(test_file_loc, 'r')
for ele in f.readlines():
    path= ele.strip()
    gray = mpimg.imread(path)
    gray = rgb2gray(gray)
    gray = cv2.resize(gray, (80,80), interpolation = cv2.INTER_AREA)
    x_valid.append(gray.flatten())
x_valid = np.array(x_valid)
x_valid = np.hstack((np.ones((x_valid.shape[0], 1)), x_valid))
f.close()


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)

sigmoid = lambda x : (1 / (1 + np.exp(-x)))

cov_mat = calc_cov_mat(x_train)
cov_mat.shape

evals, evecs = np.linalg.eig(cov_mat)

idx = evals.argsort()[::-1]   
eigenValues = evals[idx]
eigenVectors = evecs[:,idx]

target_var = .9
tot_var = np.sum(eigenValues)
perc = []
csum=0
comps = 0
tcomps=0
for ind in range(250):
    csum += eigenValues[ind]
    comps += 1
    perc.append(csum/tot_var)
    if np.real(csum/tot_var) >= target_var:
        # print(target_var*100,"% variance achieved at ",comps," components")
        tcomps=comps
        break

trans_mat = np.real(eigenVectors[:,0:tcomps])
x_train_pca = np.dot(x_train, trans_mat)
x_valid_pca = np.dot(x_valid, trans_mat)

scaler = MinMaxScaler()
x_train_pca = scaler.fit_transform(x_train_pca)
x_valid_pca = scaler.transform(x_valid_pca)


weights = np.ones((x_train_pca.shape[1], 1))
mod_y_train = pd.get_dummies(y_train)
tol = 0.000001
alpha = 0.05
epsilon = 1e-7
m = x_train_pca.shape[0]
for cind in range(len(set(y_train))):
    maj_class = mod_y_train.columns[cind]
    # print("Training classifier for class ",maj_class)
    thetas = (np.random.rand(x_train_pca.shape[1], 1) - .5)
    ly_train = np.array(mod_y_train.iloc[:,cind]).reshape((-1,1))
    error = 10*tol
    pcost = 0
    iters=0
    while error >= tol and iters <= 10000:
        hypo = np.dot(x_train_pca, thetas)
        hsig = (sigmoid(hypo))
        y1cost = np.dot(ly_train.transpose(), np.log(hsig + epsilon))
        y0cost = np.dot((1 - ly_train).transpose(), np.log(1 - hsig + epsilon))
        cost = (-1/m)*( y1cost + y0cost )

        error = abs(pcost - cost)
        pcost = cost
        thetas -= (alpha/m)*np.dot(x_train_pca.transpose(), (hsig - ly_train))
        iters += 1
    cost = cost[0][0]
    # print("Iterations = ",iters," Cost = ",cost)
    weights = np.hstack((weights, thetas))
weights = weights[:,1:] 

y_pred = []
res = sigmoid(np.dot(x_valid_pca, weights))
# print(res)
for row in range(res.shape[0]):
    ind = np.argmax(res[row,:])
    y_pred.append(mod_y_train.columns[ind])
for ele in y_pred:
    print(ele)