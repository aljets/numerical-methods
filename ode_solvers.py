#ode_solvers

import numpy as np
import copy as copy

# Solving systems of ordinary differential equations: Explicit Euler, Implicit Euler, Runge-Kutta

# Explicit Euler
# Given a function in matrix form, h step size, t_f final time, and initial y
def explicit_euler(function,h,t_f,y_init):
    t_i,t = 0,0
    n = (t_f-t_i)/h
    ylist = np.zeros((n+1,y_init.shape[0]))
    y_k = copy.copy(y_init)
    for k in range(int(n)+1):
        ylist[k] = y_k.T
        y_kplus1 = y_k + np.dot(h,function(y_k))
        t += h
        y_k = y_kplus1
    return ylist
    
# Implicit Euler
# Given a function in matrix form, h step size, t_f final time, initial y, and J Jacobian

def implicit_euler(function,h,t_f,y_init,J):
    m = function(y_init).shape[0]
    epsilon = 1E-6 # Achieve this minimum
    t_i,t = 0,0
    n = (t_f-t_i)/h
    ylist = np.zeros((n+1,y_init.shape[0]))
    y_k = copy.copy(y_init)
    y_kplus1 = copy.copy(y_init)
    for k in range(int(n)):
        ylist[k] = y_k.T
        t += h
        # Here, I solve psi(y_(k+1)) = 0 via Newton's to obtain y_(k+1):
        for i in range(1000):
            if i >= 998:
                print "MAX RANGE!"
            psi = y_kplus1 - y_k - h*function(y_kplus1)
            if np.linalg.norm(psi) <= epsilon: # If min is reached
                y_k = y_kplus1
                break
            left = np.identity(m) - h*J(y_kplus1)
            dy = np.linalg.solve(left,-psi)
            y_kplus1 = y_kplus1 + dy
    ylist[k+1] = y_k.T
    ylist = np.nan_to_num(ylist)
    np.clip(ylist,-1E100,1E100,out=ylist)
    return ylist
    
# Algorithm: Runge-Kutta: Classical 4th order
# Given a function in matrix form, h step size, t_f final time, and initial y

def runge_kutta(function,h,t_f,y_init):
    t_i,t = 0,0
    n = (t_f-t_i)/h
    ylist = np.zeros((n+1,y_init.shape[0]))
    y_k = copy.copy(y_init)
    for k in range(int(n)):
        ylist[k] = y_k.T
        k_1 = function(y_k)
        k_2 = function(y_k + .5*h*k_1)
        k_3 = function(y_k + .5*h*k_2)
        k_4 = function(y_k + h*k_3)
        y_kplus1 = y_k + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
        t += h
        y_k = y_kplus1
    ylist[k+1] = y_k.T
    return ylist

# Example, assuming use of pylab
# Function definiton:
def function(y):
    y1 = y[0][0]
    y2 = y[1][0]
    return np.array([[y2],
                     [(1-y1**2)*y2 - y1]])

# Initial y
y_0 = np.array([[2],
                  [4]])

# Jacobian
def J(y):
    y1 = y[0][0]
    y2 = y[1][0]
    return np.array([[0,1],
                 [(-2*y1*y2) - 1,(1-y1**2)]])

answer = runge_kutta(function,.01,20,y_0)
time = linspace(0,20,2001)

p0, = plt.plot(time,answer[:,0],'s',markerfacecolor='blue')
p1, = plt.plot(time,answer[:,1],'s',markerfacecolor='red')
plt.xlabel("Time")
plt.ylabel("y value")
plt.legend([p0,p1],
           ["y1","y2"])
plt.title("Plot of Function")
plt.show()

p0, = plt.plot(answer[:,0],answer[:,1],'s',markerfacecolor='blue')
plt.xlabel("y1")
plt.ylabel("y2")
plt.legend([p0],
           ["y2 vs y1"])
plt.title("Plot of Function")
plt.show()
