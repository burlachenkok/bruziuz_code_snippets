import tensorflow as tf

a = tf.add(3, 5)
print(a)
print(type(a))

# sess = tf.Session()
# print(sess.run(a))
# sess.close()

with tf.Session() as sess:
	print(sess.run(a))
