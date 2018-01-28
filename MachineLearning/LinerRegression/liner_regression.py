import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from LinerRegression import data_init

# Training Data
train_x, train_y = data_init.data_init()
x_size = len(train_x)

# Graph Input
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Set model parm
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Liner regression
prediction = tf.add(tf.multiply(x, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(prediction - y, 2)) / (2 * x_size)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        for (tx, ty) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={x: tx, y: ty})

        if (i+1) % 50 == 0:
            c = sess.run(cost, feed_dict={x: train_x, y: train_y})
            print("Time:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    # display
    plt.plot(train_x, train_y, 'ro', label='original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label="prediction line")
    plt.legend()
    plt.show()
