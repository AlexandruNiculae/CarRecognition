#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
from data import Data

img_width = 300
img_height = 300
iterations = 150
batch_step = 20
num_classes = 196

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, img_width * img_height])
x_image = tf.reshape(x, [-1, img_width,img_height, 1])

y_ = tf.placeholder(tf.float32, [None, num_classes])

# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1) # 150 x 150
#
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2) # 75 x 75
#
#
# W_fc1 = weight_variable([75 * 75 * 64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 75*75*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#
# W_fc2 = weight_variable([1024, num_classes])
# b_fc2 = bias_variable([num_classes])
#
# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

W_conv1 = weight_variable([3,3,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

W_conv2 = weight_variable([3,3,32,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2) + b_conv2)

h_pool1 = max_pool_2x2(h_conv2) # 150x 150

keep_prob1 = tf.placeholder(tf.float32)
h_drop1 = tf.nn.dropout(h_pool1, keep_prob1)

W_conv3 = weight_variable([3,3,32,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_drop1,W_conv3) + b_conv3)

h_pool2 = max_pool_2x2(h_conv3) # 75 x 75

keep_prob2 = tf.placeholder(tf.float32)
h_drop2 = tf.nn.dropout(h_pool2, keep_prob2)

h_drop2_flat = tf.reshape(h_drop2, [-1, 75*75*64])

W_fc1 = weight_variable([75 * 75 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_drop2_flat, W_fc1) + b_fc1)

keep_prob3 = tf.placeholder(tf.float32)
h_drop3 = tf.nn.dropout(h_fc1, keep_prob3)

W_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])
y_conv = tf.matmul(h_drop3, W_fc2) + b_fc2

#y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dataset = Data()
data_tuple = dataset.getTensorflowDataset()
train_data = (data_tuple[0], data_tuple[1])
test_data = (data_tuple[2],data_tuple[3])



def nextBatch(step):
	global batch_idx
	tmp = batch_idx
	batch_idx += step
	if batch_idx > len(train_data[0]):
		batch_idx = len(train_data[0])

	to_return = (train_data[0][tmp:batch_idx],train_data[1][tmp:batch_idx])

	if batch_idx == len(train_data[0]):
		batch_idx = 0

	return to_return

batch_idx = 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(iterations):
		#batch = mnist.train.next_batch(50)
		batch = nextBatch(batch_step)
		if i % 2 == 0:
			train_accuracy = accuracy.eval(feed_dict={
	  			x: batch[0], y_: batch[1], keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob1: 0.5, keep_prob2: 0.5, keep_prob3: 0.5})

	print('test accuracy %g' % accuracy.eval(feed_dict={
		#x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
		x: test_data[0], y_: test_data[1],  keep_prob1: 1.0, keep_prob2: 1.0, keep_prob3: 1.0}))
