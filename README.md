# pythonic_lbfgs
implementation of LBFGS using python code, mostly if you want to play around with the lbfgs algorithm. Most of this is based off of the book: J. Nocedal and S. J. Wright: Numerical Optimization, second edition, Springer Verlag, Berlin, Heidelberg, New York, 2006


Requirements:
autograd
numpy

Includes an implementation using sparse matrices, in case the number of degrees of freedom is really large $>1000$. (You don't want to use a fully dense Hessian matrix, which is going to be diagonal anyways for the problem)

