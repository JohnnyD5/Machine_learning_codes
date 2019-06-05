# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:00:29 2019

@author: zding5
"""
import numpy as np
# it is helpful to monitor the convergence by computing the cost J(theta)
# This section is to code J(theta) so you can check the convergence
# of the gradient descent implementation

# J(θ_0,θ_1 )=1/2m ∑_(i=1)^m((h_θ (x_i )-y_i )^2) 
# h_theta(X) = X@Theta
def cost(X,y,theta):
    m = len(X)
    return np.sum((X@theta-y)**2)/(2*m)