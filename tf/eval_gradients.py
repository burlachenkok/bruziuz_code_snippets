import tensorflow as tf

x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2

grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(grad_z)) # >> [768.0, 32.0]

# create an optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# compute the gradients for a list of variables.
# grads_and_vars = optimizer.compute_gradients(loss, <list of variables>)
grads_and_vars = optimizer.compute_gradients(z, [x,y])

# grads_and_vars is a list of tuples (gradient, variable).
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # >> [768.0, 32.0]
    print(sess.run(grads_and_vars))

