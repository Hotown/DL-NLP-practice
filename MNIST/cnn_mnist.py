import tensorflow as tf

# 测试数据导入
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 模型建立 28*28的图
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


######################## 卷积  和  池化 #############################
#
# 卷积层
# strides = [batch, in_height, in_width, in_channels]
# 步长矩阵, batch, in_channels都为1, 表示在一个样本的一个通道上的特征图上移动
# in_height, in_width 是卷积核在特征图的高度和宽度上移动的步长
#
# padding = 'SAME' 以0填充边缘，且左上和右下补0的个数相同或少一个
# padding = 'VALID' 不填充，多余的丢弃
#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
#
#
# strides和padding和卷积层相同
#
# ksize = [1, height, width, 1] 一般不在batch和channels上做池化，所以设为1
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
# 5*5的感受视野，得到32个feature，1表示1个输入通道
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 将n*784转化成一个4d向量，第2，3维表示图片宽高，第4维表示通道数（灰度图为1，RGB图为3）
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# 5个输入通道，每个5*5的感受视野得到64个feature
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

######################## 连    接    层 #############################
# 密集连接层
#
# 加入1024个神经元的全连接层，并将7*7*64的tensor reshape成向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout 减少过拟合
# 用一个placeholder代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

######################## 输    出    层 #############################
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

######################## 模型训练和评估 #############################
sess = tf.InteractiveSession()

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

# 训练
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 每100次迭代输出一次
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # keep_prob用来控制dropout的概率
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 评估
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
