#!/usr/bin/env python3

import tensorflow as tf
import sys

# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
  # compute the value of b given a is 15
  print(sess.run(b, feed_dict={a: 15}))				# >> 45

