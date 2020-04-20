# -*- coding: utf-8 -*-
"""
CS520 Spring 2020
Class Project
Authors: Marcelo Souza, Poyraz Bozkurt
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import time
from scipy.optimize import minimize
from scipy.optimize import check_grad
import cvxopt
#tf.compat.v1.disable_eager_execution()


# Energy 

def point_energy(v1, v2, alpha = 2.):
    return la.norm(v2 - v1) ** (-alpha)

def total_energy(points, alpha = 2.):
    e = 0.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
                e = e + point_energy(points[i], points[j], alpha)
    return e


# Normalized energy - does not depend on vectors' lengths

def normalized_energy(v1, v2, alpha = 2.):
    n1 = la.norm(v1)
    n2 = la.norm(v2)
    return (la.norm((v2 / n2) - (v1 / n1)) ** (-alpha))

def normalized_total_energy(points, alpha = 2.):
    e = 0.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
                e = e + normalized_energy(points[i], points[j], alpha)
    return e


# Gradient

def gradient(v, u):
    dif = u - v
    return dif * 2. * la.norm(dif) ** (-4)

def total_gradient(points):
    grads = []
    for i, p in enumerate(points):
        g = p - p           # zero vector
        for j, q in enumerate(points):
            if i != j:
                g = g + gradient(p, q)
        grads.append(g)
    return grads


# Serialization
# Map a set of N points in S^3 to a point in R^3N

def serialize(points):
    return np.hstack(points)

def deserialize(serial_points):
    rows = []
    for i in range(int(len(serial_points) / 3)):
        rows.append(serial_points[3*i:3*i+3])
    return np.vstack(rows)

def serial_total_energy(serial_points):
    # to do: implement appropriate routine
    return total_energy(deserialize(serial_points))

def serial_normalized_total_energy(serial_points):
    # to do: implement appropriate routine
    return normalized_total_energy(deserialize(serial_points))

def serial_gradient(serial_points):
    grads = []
    for i in range(int(len(serial_points) / 3)):
        v = serial_points[3*i:3*i+3]
        g = [0., 0., 0.]
        for j in range(int(len(serial_points) / 3)):
            if j != i:
                u = serial_points[3*j:3*j+3]
                g = g + gradient(v, u)
        grads.append(g)
    return np.hstack(grads)


# Hessian

def serial_hessian(serial_points, verbose=False):
    H = []
    for i in range(int(len(serial_points) / 3)):
        p = serial_points[3*i:3*i+3]
        for di in range(3):
            # build one row of the Hessian
            row = []
            msg = []
            for j in range(int(len(serial_points) / 3)):
                if (i==j):
                    for dj in range(3):
                        # sum over all q != p   [optimize: this is the negative of the sum of the else section]
                        s = 0.
                        for k in range(int(len(serial_points) / 3)):
                            if k != i:
                                q = serial_points[3*k:3*k+3]
                                dif = q - p
                                norm = la.norm(dif)
                                s = s + (16. * norm ** (-6)) * (q[dj] - p[dj]) * (q[di] - p[di])
                                if dj == di:
                                    s = s - (2. * norm ** (-4))
                        row.append(s)
                        msg.append('({:1d},{:1d})x({:1d},{:1d})'.format(i,di,j,dj))
                else:
                    q = serial_points[3*j:3*j+3]
                    dif = q - p
                    norm = la.norm(dif)
                    for dj in range(3):
                        s = (16. * norm ** (-6)) * (q[dj] - p[dj]) * (q[di] - p[di])
                        if dj == di:
                            s = s + (2. * norm ** (-4))
                        row.append(s)
                        msg.append('({:1d},{:1d})x({:1d},{:1d})'.format(i,di,j,dj))
            H.append(row)
            if verbose: print(msg)
    
    return np.array(H)


# Random points in 3D - initial set

d = 3
n = 20
alpha = 2.
#M = 2. * np.pi / n

points = []
for i in range(n):
    p = np.random.normal(size = d)
    p = p / la.norm(p)
    points.append(p)
serial_points = serialize(points)


# Checking gradients
    
check_grad(serial_total_energy, serial_gradient, serial_points)


# Trust Region

g = serial_gradient(serial_points, True)
h = serial_hessian(serial_points, True)

'trust-ncg'     # Newton conjugate gradient
'trust-krylov'  # Krylov

#method = 'trust-krylov'
method = 'trust-ncg'

print('\nprocess start')
print('method = ' + method)
print('#points = {:4d}'.format(n))
print('#steps  = {:4d}'.format(minimization_steps))
start_time = time.time()
opt = minimize(serial_normalized_total_energy, serial_points, method='trust-ncg', jac=serial_gradient, hess=serial_hessian)
end_time = time.time()
energy_decay.append(E)
print('final energy {:1.4f}'.format(energy.numpy()))
print('execution time {:8.0f} sec'.format(end_time - start_time))

if opt['success']:
    print('optimization ended with success')
else:
    print('optimization ended with failure')

opt_points = deserialize(opt['x'] )
for i in range(opt_points.shape[0]):
    opt_points[i] = opt_points[i] / la.norm(opt_points[i])
e0 = total_energy(points)
e1 = total_energy(opt_points)
print('initial points energy: {:1.4f}\nfinal energy: {:1.4f}'.format(e0, e1))


# Adam Optimization (using TensorFlow)

minimization_steps = 500
print_at_each = 10
energy_decay = []


# --- central loop ---

vertices = [tf.Variable(p) for p in points]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

print('\nprocess start')
print('\nmethod = adam')
print('#points = {:4d}'.format(n))
print('#steps  = {:4d}'.format(minimization_steps))
print('initial energy {:1.4f}'.format(total_energy(points)))
start_time = time.time()
E = []
for step in range(minimization_steps):
    with tf.GradientTape() as t:
        edges = []
        N = len(points)
        for i in range(N):
            for j in range(i + 1, N):
                edges.append(tf.subtract(vertices[i], vertices[j]))
        energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])
    
    gradients = t.gradient(energy, vertices)    # <-- choose TensorFlow gradients
    #gradients = total_gradient(vertices)        # <-- choose own gradients
    opt.apply_gradients(zip(gradients, vertices))
    vertices = [tf.Variable(v / la.norm(v.numpy())) for v in vertices]
    
    e = energy.numpy()
    #e = normalized_total_energy([v.numpy() for v in vertices])
    E.append(e)
    if print_at_each != 0:
        if (step + 1) % print_at_each == 0:
            print('step {:4d}   energy {:1.4f}'.format(step + 1, e))
    if e < 5955: break

end_time = time.time()
energy_decay.append(E)
print('final energy {:1.4f}'.format(energy.numpy()))
print('execution time {:8.0f} sec'.format(end_time - start_time))

# --- end of central loop ---



# Convex optimization

from cvxopt import solvers, matrix
# generates S_k+1 from S_k
# S_k minimizes energy conditional on sum(s in S_k)[ sum(t in S_k+1)[|<s,t>|] ] = constant
# S_k is normalized
# check https://cvxopt.org/userguide/solvers.html#s-cp
from numpy import array
def iterate_S(S):
    C = len(S) + .000001
    def F(x=None, z=None):
        if x is None:
            return 1, matrix(S)     # 1 constraint, x0 = previous set
        if np.outer(S, x).sum().sum() > C:
            return None
        
        _x = array(x)[:,0]  # converting to numpy
        f = np.array([serial_total_energy(x), np.outer(S, x).sum().sum()]).T
        Df = [serial_gradient(_x)]
        # gradient of constraint
        d = [np.sum([S[i] for i in range(len(S)) if i % 3 == 0]),
             np.sum([S[i] for i in range(len(S)) if i % 3 == 1]),
             np.sum([S[i] for i in range(len(S)) if i % 3 == 2])]
        D = np.repeat(d, int(len(S) / 3))
        D = D * np.sign(S)
        Df.append(D)
        Df = np.array(Df)
        Df = matrix(Df)     # back to cvxopt type
        
        if z is None: return f, Df
        H = matrix(serial_hessian(S))
        
        print(f)
        print(Df)
        return f, Df, H
    S_next = solvers.cp(F)
    
    # normalize S_next
    # ...
    
    return S_next


# The idea is to iterate in a loop
while False # ...
    S = iterate(S)
