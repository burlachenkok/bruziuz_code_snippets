import tensorflow as tf

#tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

a = tf.constant([2, 2], name='a')

b = tf.constant([[0, 1], 
                 [2, 3]], name='b')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# Broadcasting similar to NumPy
# Perfomr elemetwise multiplication
with tf.Session() as sess:
    print(sess.run( tf.multiply(a, b, name='mul') ))

# Creates a tensor of shape and all elements will be zeros
# These are more compact than constants in the graph def, resulting in faster startup
# (especially in distributed where the graph must be send to all workers)
with tf.Session() as sess:
    print(sess.run( tf.zeros([2, 3], tf.int32) ))

Z = tf.zeros([2, 3], tf.int32)
# for item in dir(Z): print(item)

with tf.Session() as sess:
    print(type(sess.run( tf.fill([2,3], 12, name='None') )))

with tf.Session() as sess:
    print(sess.graph.as_graph_def())

#s = tf.Session()

#print(s.run(tf.lin_space(10.0, 13.0, 4)))

# create variables with tf.Variable
s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([784,10]))

# create variables with tf.Variable
s = tf.Variable(2, name="scalar") 
m = tf.Variable([[0, 1], [2, 3]], name="matrix") 
W = tf.Variable(tf.zeros([784,10]))

# create variables with tf.get_variable
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())


# tf.Variable holds several ops:
#x = tf.Variable(...) 

#x.initializer # init op
#x.value() # read op
#x.assign(...) # write op
