# -*- coding: utf-8 -*-
"""
Class Project - CS520

@authors: Marcelo Souza, Poyraz Bozkurt
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


# Gradient

# to be applied to v, relative to u
# to do: convert to TensorFlow function
           
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


# Testing gradient function
    
p = np.random.normal(size = 2)
p = p / la.norm(p)
x1 = p

p = np.random.normal(size = 2)
p = p / la.norm(p)
x2 = p

G = gradient(x1, x2)
g1 = G
g2 = -G
_x1 = x1 - g1
_x1 = _x1 / la.norm(_x1)
_x2 = x2 - g2
_x2 = _x2 / la.norm(_x2)

G = gradient(_x1, _x2)
_g1 = G
_g2 = -G
__x1 = _x1 - _g1
__x1 = __x1 / la.norm(__x1)
__x2 = _x2 - _g2
__x2 = __x2 / la.norm(__x2)

G = gradient(_x1, _x2)
__g1 = G
__g2 = -G

print('distance between points {:1.2f}'.format(la.norm(x1 - x2)))
print('gradient norm {:1.2f}'.format(la.norm(g1)))

circle = Circle((0., 0.), radius = 1., fill = False)
fig = pl.figure()
ax = fig.add_subplot(111)
ax.axis('equal')
ax.add_patch(circle)
ax.scatter(0., 0., color='black')
ax.scatter(x1[0], x1[1], color='blue')
ax.scatter(x2[0], x2[1], color='red')
ax.arrow(x1[0], x1[1], g1[0], g1[1])
ax.arrow(x2[0], x2[1], g2[0], g2[1])
ax.scatter(_x1[0], _x1[1], marker='^', color='blue')
ax.scatter(_x2[0], _x2[1], marker='^', color='red')
ax.arrow(_x1[0], _x1[1], _g1[0], _g1[1])
ax.arrow(_x2[0], _x2[1], _g2[0], _g2[1])
ax.scatter(__x1[0], __x1[1], marker='s', color='blue')
ax.scatter(__x2[0], __x2[1], marker='s', color='red')
ax.arrow(__x1[0], __x1[1], __g1[0], __g1[1])
ax.arrow(__x2[0], __x2[1], __g2[0], __g2[1])


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

energy_decay = []


# Optimization

minimization_steps = 2000
print_at_each = 10

# --- central loop ---

vertices = [tf.Variable(p) for p in points]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

print('\nprocess start')
print('#points = {:4d}'.format(n))
print('#steps  = {:4d}'.format(minimization_steps))
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

end_time = time.time()
energy_decay.append(E)
print('final energy {:1.4f}'.format(energy.numpy()))
print('execution time {:8.0f} sec'.format(end_time - start_time))

# --- end of central loop ---


# Show energy decay

for E in energy_decay:
    pl.plot(E)


# Show sphere with random and optimized points

opt_points = [v.numpy() for v in vertices]

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

for p in points:
    ax.scatter(p[0], p[1], p[2], color='blue')
for p in opt_points:
    ax.scatter(p[0], p[1], p[2], color='red')

# surface to help visualization
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 1000)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
elev = 10.0
rot = 80.0 / 180 * np.pi
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='lightgrey', linewidth=0, alpha=0.5)
pl.show()


# Normalized energy - does not depend on vectors' lengths

def point_energy(v1, v2, alpha = 2.):
    return la.norm(v2 - v1) ** (-alpha)

def total_energy(points, alpha = 2.):
    e = 0.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
                e = e + point_energy(points[i], points[j], alpha)
    return e

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


# tests
#normalized_total_energy(points)
#normalized_total_energy([np.random.random() * p for p in points])
#normalized_total_energy(opt_points)
#normalized_total_energy([np.random.random() * p for p in opt_points])


# Serialization
# Map a set of N points in S^3 to a point in R^3N

def serialize(points):
    return np.hstack(points)

def deserialize(serial_points):
    rows = []
    for i in range(int(len(serial_points) / 3)):
        rows.append(serial_points[3*i:3*i+3])
    return np.vstack(rows)

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

def serial_total_energy(serial_points):
    # to do: implement appropriate routine
    return total_energy(deserialize(serial_points))

def serial_normalized_total_energy(serial_points):
    # to do: implement appropriate routine
    return normalized_total_energy(deserialize(serial_points))


# Convex optimization
    



# Checking gradients
    

d = 3
n = 3
alpha = 2.
#M = 2. * np.pi / n

points = []
for i in range(n):
    p = np.random.normal(size = d)
    p = p / la.norm(p)
    points.append(p)
    
    
G = [gradient(points[0], points[1]) + gradient(points[0], points[2]),
     gradient(points[1], points[0]) + gradient(points[1], points[2]),
     gradient(points[2], points[0]) + gradient(points[2], points[1])]

G    
total_energy(points)

circle = Circle((0., 0.), radius = 1., fill = False)
fig = pl.figure()
ax = fig.add_subplot(111)
ax.axis('equal')
ax.add_patch(circle)
ax.scatter(0., 0., color='black')
ax.scatter(points[0][0], points[0][1], color='red')
ax.scatter(points[1][0], points[1][1], color='green')
ax.scatter(points[2][0], points[2][1], color='blue')

serial_points = serialize(points)
    
np.hstack(total_gradient(points))
serial_gradient(serial_points)

check_grad(total_energy, total_gradient, points)
total_energy(points)
total_gradient(points)

serial_gradient(serial_points)

check_grad(serial_total_energy, serial_gradient, serial_points)

def energy_p0(p0):
    _points = points.copy()
    _points[0] = p0
    return total_energy(_points)
p0 = points[0]
energy_p0(p0)

def grad_p0(p0):
    _points = points.copy()
    _points[0] = p0
    return total_gradient(_points)
grad_p0(p0)

check_grad(energy_p0, grad_p0, p0)
check_grad(total_energy, total_gradient, points)

grad = total_gradient(points)
epsilon = .0001
_points = points.copy()
for i in range(len(_points)):
    _points[i] = _points[i] - epsilon * grad[i]
total_energy(points)
total_energy(_points)


def f_grad(serial_points):
    return serial_gradient(serial_points)[0]
def f_hess(serial_points):
    return serial_hessian(serial_points)[0]
check_grad(total_energy, f_hess, serial_points)
check_grad(f_grad, f_hess, serial_points)


# Trust Region

serial_points = serialize(points)
g = serial_gradient(serial_points, True)
h = serial_hessian(serial_points, True)


_points = points.copy()
minimize(normalized_total_energy, _points, method='trust-ncg', jac=total_gradient, hess=group_hessian)
minimize(normalized_total_energy, _points, method='trust-ncg', jac=total_gradient, hess=group_hessian)

serial_normalized_total_energy(serial_points)
minimize(serial_normalized_total_energy, serial_points, method='trust-ncg', jac=serial_gradient, hess=serial_hessian)

group_gradient(points)
group_hessian(points)
np.array(h)


def direct_serial_gradient(serial_points):
    grads = []
    for i in range(int(len(serial_points) / 3)):
        p = serial_points[3*i:3*i+3]
        for d in range(3):
            g = 0.
            for j in range(int(len(serial_points) / 3)):
                if j != i:
                    q = serial_points[3*j:3*j+3]
                    dif = q - p
                    g += dif[d] * 2. * la.norm(dif) ** (-4)
            grads.append(g)
    return np.array(grads)

    
serial_points = serialize(points)
serial_gradient(serial_points)
direct_serial_gradient(serial_points)



serial_total_energy(serial_points)
epsilon
_serial_points = serial_points.copy()
_serial_points[0] = _serial_points[0] + epsilon





serial_total_energy(_serial_points) - serial_total_energy(serial_points)


deserialize(serial_points)
deserialize(_serial_points)




d = 3
n = 500
alpha = 2.
#M = 2. * np.pi / n

points = []
for i in range(n):
    p = np.random.normal(size = d)
    p = p / la.norm(p)
    points.append(p)
serial_points = serialize(points)

#method = 'trust-krylov'
method = 'trust-ncg'

print('\nprocess start')
print('method = ' method)
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



# repeated - delete
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')

for p in points:
    ax.scatter(p[0], p[1], p[2], color='blue')
for p in opt_points:
    ax.scatter(p[0], p[1], p[2], color='red')

# surface to help visualization
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 1000)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
elev = 10.0
rot = 80.0 / 180 * np.pi
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='lightgrey', linewidth=0, alpha=0.5)
pl.show()
