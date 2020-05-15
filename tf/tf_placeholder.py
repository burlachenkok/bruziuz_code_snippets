#!/usr/bin/env python3

import tensorflow as tf
import sys

# shape=None means that tensor of any shape will be accepted as value for placeholder.
# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # short for tf.add(a, b)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # We can feed_dict any tensors. placeholders are just a way to indicate that sth must be fed.
    # It's extremely helpful for testing
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
