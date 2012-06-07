#qr_factorization
import numpy as np
import copy as copy

# Gram-Schmidt QR factorization
# The following algorithm intakes a matrix a and outputs Q and R such that A=QR, where R is upper triangular

def gramschmidt(a):
    n = a.shape[0] 
    q = np.zeros((n,n))
    r = np.zeros((n,n))
    u = np.zeros((n,n))
    for j in range(0,n):
        for i in range(0,j):
            q_t = q.transpose()
            r[i,j] = np.dot(q_t[i,:],a[:,j])
        for k in range(0,j):
            u[:,j] = u[:,j] + r[k,j]*q[:,k]
        u[:,j] = a[:,j] - u[:,j]
        r[j,j] = np.linalg.norm(u[:,j])
        q[:,j] = u[:,j]/r[j,j]
    return (q,r)

# Householder QR factorization
# The following algorithm intakes a matrix a and outputs Q and R such that A=QR, where R is upper triangular
def householder(a):
    n = a.shape[0] 
    m = a.shape[1]
    abar = copy.copy(a)
    q = np.identity(n)
    r = np.zeros((m,m),dtype=np.float64)
    u = np.zeros((n,n),dtype=np.float64)
    for k in range(0,m-1):
        x = abar[k:n,k]
        gamma = -np.sign(x[0])*np.sqrt(np.dot(x.transpose(),x))
        epsilon = np.identity(n-k)
        v = (x - gamma*epsilon[:,0])
        beta = -2/np.dot(v,v)
        w = np.dot(v,abar[k:n,k:n])
        abar[k:n,k:m] = abar[k:n,k:m] + beta*np.dot(np.transpose(v[np.newaxis]),w[np.newaxis])
    r = abar[0:m,0:m]
    for k in reversed(range(0,m-1)):
        abar2 = copy.copy(a)
        for l in range(0,k+1):
            x_q = abar2[l:n,l]
            gamma_q = -np.sign(x_q[0])*np.sqrt(np.dot(x_q.transpose(),x_q))
            epsilon_q = np.identity(n-l)
            v_q = (x_q - gamma_q*epsilon_q[:,0])
            beta_q = -2/np.dot(v_q,v_q)
            w_q = np.dot(v_q,abar2[l:n,l:n])
            abar2[l:n,l:m] = abar2[l:n,l:m] + beta_q*np.dot(np.transpose(v_q[np.newaxis]),w_q[np.newaxis])
        u = np.dot(v_q,q[k:n,k:n])
        q[k:n,k:n] = q[k:n,k:n] + beta_q*np.dot(np.transpose(v_q[np.newaxis]),u[np.newaxis])
    return (q,r)
