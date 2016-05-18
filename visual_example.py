import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.name_scope('data'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    # tf.scalar_summary('data/x', x)
    # tf.scalar_summary('data/y_', y_)

with tf.name_scope('hidden'):
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    product = tf.nn.softmax(tf.matmul(x, W) + b, name='product')
    # tf.scalar_summary('hidden/W', W)
    # tf.scalar_summary('hidden/b', b)
    # tf.scalar_summary('hidden/product', product)

with tf.name_scope('xentropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(product), reduction_indices=[1]), name='loss')
    # tf.scalar_summary('xentropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()


sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

writer = tf.train.SummaryWriter("/tmp/test", sess.graph)
