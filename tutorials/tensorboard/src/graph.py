import tensorflow as tf
import datetime


logdir = '../logger/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '- computational_graph'
with tf.compat.v1.Session() as sess:

    # Build a graph.
    x = tf.compat.v1.Variable(5, name='x')
    y = tf.compat.v1.Variable(10, name='y')
    ten = tf.constant(11, name='ten')

    f = tf.add(tf.multiply(x,y, 'multiply'), ten)
    
    writer = tf.compat.v1.summary.FileWriter(logdir, sess.graph)
    
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    result = sess.run(f)
    print(result) # --> 60
