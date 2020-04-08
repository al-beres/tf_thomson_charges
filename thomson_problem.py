# -*- coding: utf-8 -*-
"""
Class Project - CS520

@authors: Marcelo Souza, Poyraz Bozkurt
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import time
import tensorflow as tf

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
def show(points, color = 'blue'):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in points:
        ax.scatter(p[0], p[1], p[2], color=color)


# TensorFlow 2.1 environmnent
        
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


# Main script

d = 3
n = 5
alpha = 3.

points = initial_set(n, d)

vertices = [tf.Variable(p) for p in points]
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

minimization_steps = 1000
print_at_each = 100
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
    vertices = vertices / la.norm(vertices[0].numpy())

    if step % print_at_each == 0:
        print('step {:2d}   energy {:1.8f}'.format(step, energy.numpy()))



vertices[0].assign(vertices[0] / 2.)
edges


op = adam.minimize(energy, vertices)


op = adam.minimize(tf_energy, tf_vertices)


show(vertices, 'blue')



with tf.GradientTape() as t:
    energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in edges])

    #### Option 1
    
    # Is the tape that computes the gradients!
    gradients = t.gradient(energy, vertices)
    # The optimize applies the update, using the variables
    # and the optimizer update rule
    opt.apply_gradients(zip(gradients, vertices))
    









show(points, 'blue')
show(opt_points, 'red')












# Performance measurements

print("TensorFlow / Adam / CUDA")
start = time.time()
opt_points = tf_optimize_adam(points)
end = time.time()
print(end - start)
