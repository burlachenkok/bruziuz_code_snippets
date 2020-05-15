import tensorflow as tf
import sys

a = tf.constant(2, name='a') # is stored in graph definion, value can be observed in tensor-board
b = tf.constant(3, name='b') # is stored in graph definion, value can be observed in tensor-board
x = tf.add(a, b, name='add2')

s = tf.Variable(2, name="scalar")
#print(type(s))
#sys.exit(0)

# Similar to numpy.zeros
z1 = tf.zeros([1, 1], tf.int32)
m1 = tf.fill([2, 3], 8) 

y = tf.add(x,m1)

# Similar to numpy.zeros_like
#z2 = tf.zeros_like(z1)

# Create the summary writer after graph definition and before running your session
# Specify "graph" or any location where you want to keep your event files

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    # Also possible create writer after session
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph

    # You have to initialize your variables
    sess.run(tf.global_variables_initializer())

    y_out = sess.run([y]) 
    print("Session output")
    print(y_out)

#    print("Print graph definition")
#    print(sess.graph.as_graph_def())

writer.close()
