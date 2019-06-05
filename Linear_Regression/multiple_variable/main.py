# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:38:15 2019

@author: zding5
"""
# Linear regression by gradient descent

import numpy as np
from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
from plotData import plotJHistory


#-------------------- Part 1: Setup----------------
# Load Dta
data = np.loadtxt('ex1data2.txt', delimiter = ',')
# m: number of training examples
# n: number of features
m,n = len(data),len(data[0])-1
x = data[:,:n].reshape(m,n)
y = data[:,-1].reshape(m,1)
theta = np.zeros((n+1,1))
# Print out original data and picture
print("The first ten lines from data set:")
for i in range(10):
    try: print("{0},{1}".format(x[i],y[i]))
    except: break
# computation environment setup
iteration = 1000
alpha = 0.3


#---------------------- Part 2: Feature Normalization-----------------
# Scale features and set them to zero mean
print('Normalizing Features ...\n')
x,mu,sd = featureNormalize(x,n)
X = np.hstack((np.ones((m,1)),x)) # Add a column of ones to x as x0
#
#---------------------- Part 3: Gradient Descent-----------------

# finding theta, def gradientDescent(X,y,theta,alpha,num_iter)
theta, J_history = gradientDescent(X,y,theta,alpha,iteration)
print("Expected theta value is:\n", theta)
# with ex1data1.txt, the theta should be [[-3.63029144]
# [ 1.16636235]]
#
##---------------------- Part 4: Convergence Visualization---------------
# Plot figure def plotJHistory(J_history, iteration)
plotJHistory(J_history, iteration)

#------------------------ Part 5: Estimate as validation-------------------
# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
area_norm = (1650 - mu[0]) / (sd[0])
br_norm = (3 - float(mu[1]))/float(sd[1])
house_norm_padded = np.array([1, area_norm, br_norm])

price = np.array(house_norm_padded).dot(theta)
print("the prie is:", price)
# most accurate answer should be 293081.46