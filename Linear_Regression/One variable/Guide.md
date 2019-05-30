In this code, we will implement linear regression with Batch Gradient
Descent Method. We will only consider one x variable.

Setup
=====

  Name    Code                                                 Note
  ------- ---------------------------------------------------- -----------------------------
  data    data = np.loadtxt('ex1data1.txt', delimiter = ',')   (m,1)
  m       len(data)                                            number of rows(traing sets)
  n       len(data\[0\])                                       number of columns(features)
  x       data\[:,:n-1\].reshape(m,n-1)                        without $x_{0}$, (m,n-1)
  X       np.hstack((np.ones((m,1)),x))                        with $x_{0}$, (m,n)
  y       data\[:,-1\].reshape(m,1)                            (m,1)
  theta   np.zeros((n,1))                                      (n,1)

Key Equations
=============

Hypothesis:
-----------

$$h_{\theta}\left( x \right) = \theta X = X@theta$$

Cost Function:
--------------

$$J\left( \theta_{0},\theta_{1} \right) = \frac{1}{2m}\sum_{i = 1}^{m}\left( h_{\theta}\left( x_{i} \right) - y_{i} \right)^{2} = \ np.sum((X@theta - y)**2)/(2*m)$$

Gradient Descent:
-----------------

$$\theta_{j} \theta_{j} - \alpha\frac{1}{m}\sum_{i = 1}^{m}{{((h}_{\theta}\left( x^{\left( i \right)} \right) - y^{\left( i \right)})}x_{j}^{\left( i \right)})$$

theta = theta - np.sum((<X@theta-y)*X,axis=0).reshape((n,1))*alpha/m>

Result for the ex1data1 file:
=============================

![](media/image1.png){width="3.197516404199475in"
height="0.4686909448818898in"}

![](media/image2.png){width="5.362332677165354in"
height="3.6952865266841646in"}

![](media/image3.png){width="4.945570866141733in"
height="3.3063090551181102in"}
