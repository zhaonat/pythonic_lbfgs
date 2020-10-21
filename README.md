# pythonic_lbfgs
implementation of LBFGS using python code, mostly if you want to play around with the lbfgs algorithm

Requirements:
autograd
numpy

Includes an implementation using sparse matrices, in case the number of degrees of freedom is really large $>1000$. (You don't want to use a fully dense Hessian matrix, which is going to be diagonal anyways for the problem)

