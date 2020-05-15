import tensorflow as tf

a = tf.Variable(2, name='a')
b = tf.Variable(3, name='a')
x = tf.add(a, b)

# Create the summary writer after graph definition and before running your session
# Specify "graph" or any location where you want to keep your event files

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
    print(sess.run(tf.global_variables_initializer()))

    # Also possible create writer after session
    # writer = tf.summary.FileWriter('./graphs', sess.graph) # if you prefer creating your writer using session's graph
    out = sess.run(x)
    print(out)
    print(type(out))

writer.close()
