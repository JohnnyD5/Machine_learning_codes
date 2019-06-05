#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:26:17 2019

@author: dingzhiz
"""
import numpy as np
# featureNormalize(x,n) returns a normalized version of X where
# the mean value of each feature is 0 and the standard deviation
# is 1. This is often a good preprocessing step to do when
# working with learning algorithms.

def featureNormalize(x,n):
    # Here x doesn't include the first column with all 1 in X
    mu = np.zeros((n,1))
    sd = np.zeros((n,1))
    for i in range(n):
        mu[i] = np.mean(x[:,i])
        sd[i] = np.std(x[:,i])
        x[:,i] = (x[:,i] - mu[i])/sd[i]
    print(mu,sd)
    return x, mu, sd