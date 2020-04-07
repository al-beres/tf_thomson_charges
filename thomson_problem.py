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
        
def tf_vertices_and_edges(points):
    N = len(points)
    vertices = tf.Variable(points, name='points')
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append(tf.math.subtract(vertices[i], vertices[j]))
    return vertices, edges
    
def tf_vertices_and_energy(points, alpha = 2.):
    N = len(points)
    vertices = tf.Variable(points)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append(tf.math.subtract(vertices[i], vertices[j]))
    energy = tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in tf_edges])
    return vertices, energy
    

# Main script

d = 3
n = 5
alpha = 3.

points = initial_set(n, d)

tf_vertices, tf_edges = tf_vertices_and_edges(points)
tf_vertices, tf_energy = tf_vertices_and_energy(points)


@tf.function
def energy():
    return tf_energy
    return tf.math.add_n([1. / tf.math.reduce_sum(tf.math.abs(e) ** alpha) for e in tf_edges])

adam = tf.keras.optimizers.Adam(learning_rate=0.001)

op = adam.minimize(energy, tf_vertices)


op = adam.minimize(tf_energy, tf_vertices)
















show(points, 'blue')
show(opt_points, 'red')












# Performance measurements

print("TensorFlow / Adam / CUDA")
start = time.time()
opt_points = tf_optimize_adam(points)
end = time.time()
print(end - start)
