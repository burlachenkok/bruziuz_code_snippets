import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3]) # a is placeholder for a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

c = a + b # use the placeholder as you would any tensor

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	# compute the value of c given the value of a is [1, 2, 3]
	print(sess.run(c, {a: [1, 2, 3]})) 		# [6. 7. 8.]

writer.close()

