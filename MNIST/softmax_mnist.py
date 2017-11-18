import tensorflow as tf

# 测试数据导入
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 模型建立
x = tf.placeholder(tf.float32, [None, 784])

# 权值矩阵和偏置数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# y'
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵作为loss function = - ∑y'log(y)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降法，学习率0.01，得到最小交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 模型训练
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 比较预测值和真实值
    # return a array of boolean
    # e.g [True, True, False, True]
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
