#!/usr/bin/env python3

import tensorflow as tf
import sys

g = tf.Graph()
with g.as_default():
    x = tf.add(3, 5)

#print(x)

with tf.Session(graph=g) as sess:
    print(sess.run(x))
    print(sess)
