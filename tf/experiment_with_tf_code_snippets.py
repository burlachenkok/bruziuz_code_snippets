#!/usr/bin/env pytho3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#=======================================================================
x = tf.constant([[1, 1, 1], [1, 1, 1]])

instructions = '''
tf.reduce_sum(x)     # 6
tf.reduce_sum(x, 0)  # [2, 2, 2]
tf.reduce_sum(x, 1)  # [3, 3]
tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
tf.reduce_sum(x, [0, 1])  # 6
'''
#=======================================================================
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('=======================================================================')
linesWithComments = instructions.splitlines()
lineNumber = 1

for cmd in linesWithComments:
    if len(cmd.replace(' ', '')) == 0:
        continue
    print(f"Execute line #{lineNumber}: {cmd}") 
    sharppos = cmd.find('#')
    if sharppos != -1:
       cmd = cmd[:sharppos]
    exec("print(' Result: ', sess.run(" + cmd + "))")
    lineNumber += 1
print('=======================================================================')

sess.close()
writer.close()
