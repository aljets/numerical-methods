#unconstrained-optimization
import numpy as np

# Unconstrained function minimization using Newton's method, exact Hessian,
# given a function and its gradient gradf, and using BFGS approximation of the Hessian

# These particular formuls are for functions of the form f(x) = 1/2 (x)^T Q x - b^T x
# where Q is symmetric positive definite

def newton_exact(q,x_0,b,gradf):
    eps_1 = 1.0*10**-4 # Find a solution less than this value
    s = 1
    sigma = 1E-4
    beta = 0.5
    n = 1000 # Set n maximum times solver can run
    x_k = x_0 # Set x equal to x_0, initially
    for k in range(0,n):
        gradf = np.dot(q,x_k) - b 
        if (k == n-1):
            print "ERROR: End of range. Adjust maximum n value!"
        if (np.linalg.norm(gradf) <= eps_1): # Finish criteria
            print "Success! Finish criteria reached at",k,"th iteration."
            break
        d = linear_solver(q,-gradf) #Solve Qd=-gradf for d where Q is exact hessian
        # Begin Armijo Rule Line Search to compute alpha
        for j in range(0,n):
            alpha = (beta**j)*s
            x_k1 = x_k + np.dot(alpha,d)
            LHS = (.5*np.dot(np.dot(x_k1.T,q),x_k1)-np.dot(b.T,x_k1))-(.5*np.dot(np.dot(x_k.T,q),x_k)-np.dot(b.T,x_k))
            RHS = sigma*alpha*np.dot(gradf.T,d)
            if (LHS <= RHS):
                break
            if (j == n-1):
                print "ERROR: End of range. Adjust maximum n value!"
        # End Armijo Rule Line Search, alpha found
        x_k = x_k + alpha*d
    return x_k    

# Minimizatio using BFGS

def bfgs_hessian(q,x_0,b):
    H = np.identity(x_0.shape[0])
    eps_1 = 1.0*10**-4 # Find a solution less than this value
    s_armijo = 1
    sigma = 1E-4
    beta = 0.5
    n = 1000 # Set n maximum times solver can run
    x_k = x_0 # Set x equal to x_0, initially
    for k in range(0,n):
        if (k == n-1):
            print "ERROR: End or range. Adjust maximum n value!"
        gradf = np.dot(q,x_k) - b 
        if (np.linalg.norm(gradf) <= eps_1): # Finish criteria
            print "Success! Finish criteria reached at",countk,"th iteration."
            break
        d = linear_solver(H,-gradf) #Solve Hd=-gradf for d
        # Begin Armijo Rule Line Search to compute alpha
        for j in range(0,n):
            alpha = (beta**j)*s_armijo
            x_k1 = x_k + np.dot(alpha,d)
            LHS = (.5*np.dot(np.dot(x_k1.T,q),x_k1)-np.dot(b.T,x_k1))-(.5*np.dot(np.dot(x_k.T,q),x_k)-np.dot(b.T,x_k))
            RHS = sigma*alpha*np.dot(gradf.T,d)
            if (LHS <= RHS):
                break
        # End Armijo Rule Line Search, alpha found
        x_prev = x_k
        x_k = x_k + alpha*d
        s = x_k - x_prev
        y = (np.dot(q,x_k)-b)-(np.dot(q,x_prev)-b)
        if (np.dot(s.T,y)>0):
            H_s = np.dot(H,s)
            H = H + np.dot(y,y.T)/np.dot(s.T,y)-np.dot(H_s,H_s.T)/np.dot(s.T,H_s)
    print "Found positive definite Hessian approximate to be"
    print H
    return x_k
