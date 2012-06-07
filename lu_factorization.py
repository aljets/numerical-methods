#lu_factorization
import numpy as numpy

# The following code solves Ax = B for the vector x by LU factorization.

# LU factorization code:
def factorize(a):
    n = a.shape[0]
    u = numpy.zeros((n,n))
    l = numpy.zeros((n,n))
    for j in range(0,n):
        for i in range(0,j+1):
            if i > 0:
                for k in range(0,i):
                    u[i,j] = u[i,j] + l[i,k]*u[k,j]
            u[i,j] = a[i,j] - u[i,j]
        for i in range(j+1,n):
            if j > 0:
                for k in range(0,j):
                    l[i,j] = l[i,j] + l[i,k]*u[k,j]
            l[i,j] = (a[i,j] - l[i,j])/u[j,j]
        l[j,j]=1
    return(u,l)

# Forward substitution on matrix l
# for matrix equation ly=b where l is lower triangular, returns y
def forward_sub(l,b):
    y = numpy.zeros((l.shape[0],1))
    for i in range(0,l.shape[0]):
        for k in range(0,i):
            y[i,0] = y[i,0] + l[i,k]*y[k,0]
        y[i,0] = (b[i,0]-y[i,0])/l[i,i]
    return y

# Backward substitution on matrix u
# for matrix equation ux=y where u is upper triangular, returns x
def backward_sub(u,y):
    n = u.shape[0]
    x = numpy.zeros((u.shape[0],1))
    for i in reversed(range(0,n)):
        for k in range(i+1,n):
            x[i,0] = x[i,0] + u[i,k]*x[k,0]
        x[i,0] = (y[i,0]-x[i,0])/u[i,i]
    return x

def linear_solver(a,b): # Solver for x in ax=b form
    u = factorize(a)[0]
    l = factorize(a)[1]
    y = forward_sub(l,b) # Solve Ly = b for y
    return backward_sub(u,y) # Solve Ux = y for x\
