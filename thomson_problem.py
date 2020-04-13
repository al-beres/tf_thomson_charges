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

#tf.compat.v1.disable_eager_execution()


# Basic functionality

# initial set of points randomly distributed
def initial_set(n, d):
    points = []
    for i in range(n):
        p = np.random.normal(size = d)
        p = p / la.norm(p)
        points.append(p)
    return points

# 2d or 3d plot
def show(points, color = 'blue', dim = 3):
    if dim == 3:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p in points:
            ax.scatter(p[0], p[1], p[2], color=color)
    else:
        circle = Circle((0., 0.), radius = 1., fill = False)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(circle)
        ax.axis('equal')
        for p in points:
            ax.scatter(p[0], p[1], color=color)


# Gradient

# to be applied to v, relative to u
# to do: convert to TensorFlow function
           
def gradient(v, u):
    dif = u - v
    return dif * 2. / la.norm(dif) ** 4

def group_gradient(points):
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

# --- central loop ---

vertices = [tf.Variable(p) for p in points]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

minimization_steps = 200
print_at_each = 10
E = []

print('\nprocess start')
print('#points = {:4d}'.format(n))
print('#steps  = {:4d}'.format(minimization_steps))
start_time = time.time()
for step in range(minimization_steps):
    with tf.GradientTape() as t:
        edges = []
        N = len(points)
        for i in range(N):
            for j in range(i + 1, N):
                edges.append(tf.subtract(vertices[i], vertices[j]))
        energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])
    
    #gradients = t.gradient(energy, vertices)    # <-- choose TensorFlow gradients
    gradients = group_gradient(vertices)        # <-- choose own gradients
    opt.apply_gradients(zip(gradients, vertices))
    vertices = [tf.Variable(v / la.norm(v.numpy())) for v in vertices]

    e = energy.numpy()
    E.append(e)
    if print_at_each != 0:
        if (step + 1) % print_at_each == 0:
            print('step {:4d}   energy {:1.4f}'.format(step + 1, e))

print('final energy {:1.4f}'.format(energy.numpy()))
end_time = time.time()
print('execution time {:8.0f} sec'.format(end_time - start_time))

energy_decay.append(E)

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
