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

# 3d plot
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


# Main script

d = 3
n = 10
alpha = 3.

points = initial_set(n, d)
show(points, 'blue')


# TensorFlow 2.1 environmnent

@tf.function
def min_energy_tf_gradients(vertices, minimization_steps, print_at_each = 0):
    for step in range(minimization_steps):
        with tf.GradientTape() as t:
            edges = []
            N = len(points)
            for i in range(N):
                for j in range(i + 1, N):
                    edges.append(tf.subtract(vertices[i], vertices[j]))
            energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])
        
        gradients = t.gradient(energy, vertices)
        opt.apply_gradients(zip(gradients, vertices))
        vertices = [tf.Variable(v / la.norm(v.numpy())) for v in vertices]
    
        if print_at_each != 0:
            if (step + 1) % print_at_each == 0:
                print('step {:4d}   energy {:1.4f}'.format(step + 1, energy.numpy()))
v1 = x1
v2 = x2
M = 1.2

# gradient to be applied to v, relative to u
#@tf.function
def gradient(v, u):
    d = u - v
    m = la.norm(d)
    return d / (m ** 3)


# testing gradient
    
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
g1 = G
g2 = -G
__x1 = _x1 - g1
__x1 = __x1 / la.norm(__x1)
__x2 = _x2 - g2
__x2 = __x2 / la.norm(__x2)

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
ax.scatter(__x1[0], __x1[1], marker='s', color='blue')
ax.scatter(__x2[0], __x2[1], marker='s', color='red')


# pure numpy experiment in 2D

d = 2
n = 30
#M = 2. * np.pi / n

points = []
for i in range(n):
    p = np.random.normal(size = d)
    p = p / la.norm(p)
    points.append(p)


n_steps = 1000
mu = .01
_p = points.copy()
for step in range(n_steps):
    for i in range(len(points)):
        g = 0.
        for j in range(len(points)):
            if i != j:
                g = g + gradient(_p[i], _p[j])
        pi = _p[i] - mu * g
        pi = pi / la.norm(pi)
        _p[i] = pi
            


circle = Circle((0., 0.), radius = 1., fill = False)
fig = pl.figure()
ax = fig.add_subplot(111)
ax.add_patch(circle)
ax.axis('equal')
for e, p in enumerate(points):
    ax.scatter(p[0], p[1], color='blue')
    ax.annotate(e, (p[0], p[1]))
for e, p in enumerate(_p):
    ax.scatter(p[0], p[1], color='red', marker='^')
    ax.annotate(e, (p[0], p[1]))
        
        
show(points, color='blue', dim = 2)
show(_p, color='red', dim = 2)


vertices = [tf.Variable(p) for p in points]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

minimization_steps = 1000
print_at_each = 100

print('\nprocess start')
print('# points = {:4d}'.format(n))
print('# steps  = {:4d}'.format(minimization_steps))
start_time = time.time()
min_energy_tf_gradients(vertices, minimization_steps, print_at_each)
print('final energy {:1.4f}'.format(energy.numpy()))
end_time = time.time()
print('execution time {:8.2f} sec'.format(end_time - start_time))

opt_points = [v.numpy() for v in vertices]


# Show

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






'''


# Performance measurements
import time

print("TensorFlow / Adam / CUDA")
start = time.time()
opt_points = tf_optimize_adam(points)
end = time.time()
print(end - start)




@tf.function
def tf_vertices_and_edges(points):
    N = len(points)
    vertices = tf.Variable(points, name='points')
    edges = []
    #for i in range(N):
    #    for j in range(i + 1, N):
    #        edges.append(tf.math.subtract(vertices[i], vertices[j]))
    return vertices, edges
    

@tf.function
def tf_vertices_and_energy(points, alpha = 2.):
    N = len(points)
    vertices = [tf.Variable(p) for p in points]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append(tf.math.subtract(vertices[i], vertices[j]))
    energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])
    return vertices, energy

@tf.function
def minimize_energy():
    

@tf.function
def energy():
    #return tf_energy
    return tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])



with tf.GradientTape() as t:
    energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])

    #### Option 1
    
    # Is the tape that computes the gradients!
    gradients = t.gradient(energy, vertices)
    # The optimize applies the update, using the variables
    # and the optimizer update rule
    opt.apply_gradients(zip(gradients, vertices))
    
'''