#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

# Create the summary writer after graph definition and before running your session
# Graphs or any location where you want to keep your event files
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# Graphs or any location where you want to keep your event files
with tf.Session() as sess:
    print(sess.run(x))
