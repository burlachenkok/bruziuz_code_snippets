#!/usr/bin/env python3

import tensorflow as tf
import sys

W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
  sess.run(W.initializer) # Initialize a single variable
  sess.run(assign_op)
  print(W.eval())         # Similar to print(sess.run(W))
  print(W)
