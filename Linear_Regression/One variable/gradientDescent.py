# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:27:32 2019

@author: zding5
"""
import numpy as np
from computeCost import cost
def gradientDescent(X,y,theta,alpha,num_iter):
    m,n = len(X),len(X[0])
    J_history = np.zeros(num_iter+1)
    J_history[0] = cost(X,y,theta)
    for i in range(1, num_iter+1):
        theta = theta - np.sum((X@theta-y)*X,axis=0).reshape((n,1))*alpha/m         # suan sum de fang shi shi bu dui de
        J_history[i] = cost(X,y,theta)
    #print("History of cost function is: ", J_history)
    return theta