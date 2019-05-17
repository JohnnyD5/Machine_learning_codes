# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:38:15 2019

@author: zding5
"""
# Linear regression by gradient descent

import numpy as np
from plotData import plotData
from gradientDescent import gradientDescent
from visual_J import visual_J

# Basic setup
data = np.loadtxt('ex1data1.txt', delimiter = ',')
m,n = len(data),len(data[0])
x = data[:,:n-1].reshape(m,n-1)
y = data[:,-1].reshape(m,1)
X = np.hstack((np.ones((m,1)),x)) # Add a column of ones to x as x0
theta = np.zeros((n,1))

# computation environment setup
iteration = 1500
alpha = 0.01

#print(np.sum((X*theta.flatten()-y)*X,axis=0)*alpha/m)


# finding theta, def gradientDescent(X,y,theta,alpha,num_iter)
theta = gradientDescent(X,y,theta,alpha,iteration)
print(r'Expected theta value is:', theta)
# Plot figure def plotData(x,y)
plotData(X,y,theta)
visual_J(X,y)