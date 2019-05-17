import numpy as np
# J(θ_0,θ_1 )=1/2m ∑_(i=1)^m((h_θ (x_i )-y_i )^2) 
# h_theta(X) = X@Theta
def cost(X,y,theta):
    m = len(X)
    return np.sum((X@theta-y)**2)/(2*m)
