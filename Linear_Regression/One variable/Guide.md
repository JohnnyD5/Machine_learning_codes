In this code, we will implement linear regression with Batch Gradient Descent Method.
We will only consider one x variable.

1 Setup:
we'll import data
Cost function
J(θ_0,θ_1 )=1/2m ∑_(i=1)^m▒(h_θ (x_i )-y_i )^2 
This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved 1/2 as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the 1/2 term. The following image summarizes what the cost function does:
 
Figure 2 Cost function handout
Gradient Descent Algorithm
The algorithm is:
Repeat until convergence
