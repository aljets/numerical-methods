#nonlinear_equations-newton

# Newton's method to solve systems of nonlinear equations
# with function F, Jacobian J, and initial guess of x_start for a range of lambda, l[i]

import numpy as np
import copy as copy

def newton(F,J,x_start,l):
    x = copy.copy(x_start)
    epsilon = 1*10**-8 # Find a solution less than this value
    results = []
    # This solves for a range of lambda (lambda is l[i])
    for i in range(len(l)):  
        for k in range(100):
            if np.linalg.norm(F(l[i],x)) <= epsilon:
                break
            d = linear_solver(J(x),-F(l[i],x))
            print x
            x += d
        if len(l) ==1:
            results = np.array([x[0][0],x[1][0]])
            break            
        solution_i = [l[i],x[0][0],x[1][0]]
        results.append(solution_i)
    return results

# Damped Newton's method using merit function (minimizing F(x) = 0 by minimizing merit
# function, phi), using the exact Jacobian of F(x)

def damped_newton(F,J,x_start,l,gradphi): # Uses exact Jacobian
    x = x_start
    epsilon = 1*10**-8 # Find a solution less than this value
    beta = 0.5
    sigma = 1E-4
    max = 50 # Maximum number of iterations
    results = []
    # This solves for a range of lambda (lambda is l[i])
    for i in range(len(l)):      
        #This is the Newton solver
        for k in range(max):
            if np.linalg.norm(F(l,x)) <= epsilon:
                print "Success at",k
                break
            # Check for singularity
            singularity = False
            a = factorize(J(x))[0]
            for row in range(J(x).shape[0]):
                for col in range(J(x).shape[0]):
                    if row == col:
                        if np.abs(a[row][col]) < epsilon:
                            print "J was singular!"
                            singularity = True
                            break
            # If singular, use steepest descent direct for the merit functin phi
            # If non-singular, solve J(x)d = -F(x) with descent direction d
            if singularity:
                d = -gradphi(l[i],x)
                # J was singular, so use this search direction
            else:
                d = linear_solver(J(x),-F(l[i],x))
                # Solves Jd = -F for descent direction d
            # End Check
            for j in range(max):
                phi_k = 0.5*np.dot(F(l[i],x).T,F(l[i],x))
                x_previous = x
                x += np.dot(beta,d)
                phi_kplus1 = 0.5*np.dot(F(l[i],x).T,F(l[i],x))
                # Criteria: merit function minimization
                if (phi_kplus1 <= phi_k + sigma*beta*np.dot(gradphi(l[i],x_previous).T,d)):
                    break
        solution_i = [l[i],x[0][0],x[1][0]]
        results.append(solution_i)
    return results

# For example,
# my constants and my lambdas I am iterating over
a_1, a_2, a_3 = 1.5, 5.0, 3.0
l_sample = np.append(arange(0,2*np.pi,0.10),2*np.pi)

#defining my function
def F_sample(l,thetas):
    theta_2 = thetas[0][0]
    theta_3 = thetas[1][0]
    return np.array([[a_1*np.cos(l) + a_2*np.cos(theta_2) + a_3*np.cos(theta_3) - 4.5],
                     [a_1*np.sin(l) + a_2*np.sin(theta_2) + a_3*np.sin(theta_3) + 3.0]])
def J_sample(thetas):
    theta_2 = thetas[0][0]
    theta_3 = thetas[1][0]
    return np.array([[-a_2*np.sin(theta_2),-a_3*np.sin(theta_3)],
                    [a_2*np.cos(theta_2),a_3*np.cos(theta_3)]])
    
def gradphi(l,thetas):
    theta_2 = thetas[0][0]
    theta_3 = thetas[1][0]
    return np.array([[(-a_2*(-4.5+a_1*np.cos(l)+a_2*np.cos(theta_2)+a_3*np.cos(theta_3))*np.sin(theta_2)+
                      2*a_2*np.cos(theta_2)*(3+a_1*np.sin(l)+a_2*np.sin(theta_2)+a_3*np.sin(theta_3)))],
                     [(-a_3*(-4.5+a_1*np.cos(l)+a_2*np.cos(theta_2)+a_3*np.cos(theta_3))*np.sin(theta_3)+
                      2*a_3*np.cos(theta_3)*(3+a_1*np.sin(l)+a_2*np.sin(theta_2)+a_3*np.sin(theta_3)))]])
        
x_initial = np.array([[4.17],
                      [0.785]])
# Solved:    
# newton_example = newton(F_sample,J_sample,x_initial,l_sample)
# damped_newton_example = damped_newton(F_sample,J_sample,x_initial,l_sample,gradphi)
