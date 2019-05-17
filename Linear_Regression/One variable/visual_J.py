# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:35:40 2019

@author: zding5
"""
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from computeCost import cost
import matplotlib.pyplot as plt
def visual_J(X,y):
    theta0 = np.linspace(-10,10,100)
    theta1 = np.linspace(-1,4,100)
    J = np.zeros((len(theta0), len(theta1)))
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            J[i,j] = cost(X,y,[[theta0[i]],[theta1[j]]])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta1,theta0,J)
    ax.set_xlabel(r'$\theta0$')
    ax.set_ylabel(r'$\theta1$')