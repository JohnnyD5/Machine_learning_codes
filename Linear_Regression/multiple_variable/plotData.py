# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:20:36 2019

@author: zding5
"""
import matplotlib.pyplot as plt
import numpy as np

def plotJHistory(J_history, iteration):
    num_iter = np.arange(1, iteration+1)
    fig, ax = plt.subplots(figsize = (6,4))
    ax.plot(num_iter,J_history,'o-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost Function')


# This plotdata function only works for one feature case
def plotData(X,y,theta):
    x = X[:,1]
    new_y = np.sum((X@theta),axis=1)
    fig, ax = plt.subplots(figsize = (6,4))
    ax.plot(x,y,'o', x,new_y)
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('Profit in $10,000s')
    return None