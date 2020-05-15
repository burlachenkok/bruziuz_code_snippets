import tensorflow as tf

a = tf.constant(2) # is stored in graph definion, value can be observed in tensor-board
b = tf.constant(3) # is stored in graph definion, value can be observed in tensor-board
x = tf.add(a, b)

# Create the summary writer after graph definition and before running your session
# Specify "graph" or any location where you want to keep your event files

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	# Also possible create writer after session
        # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
	print(type(sess.run(x)))
writer.close()
