import time

start = int(round(time.time() * 1000))
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data1', one_hot=True)
import tensorflow as tf

#use the convenient InteractiveSession class, which makes TensorFlow more flexible about how you structure your code.
#It allows you to interleave operations which build a computation graph with ones that run the graph. 
#This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession,
#then you should build the entire computation graph before starting a session and launching the graph.


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
sess.run(tf.global_variables_initializer())

#Weight Initialization
#To create this model, we're going to need to create a lot of weights and biases. 
#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Convolution and Pooling
#TensorFlow also gives us a lot of flexibility in convolution and pooling operations. 
#How do we handle the boundaries? What is our stride size? In this example, we're always going to choose the vanilla version. 
#ur convolutions uses a stride of one and are zero padded so that the output is the same size as the input. 
#Our pooling is plain old max pooling over 2x2 blocks.
#To keep our code cleaner, let's also abstract those operations into functions.

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




W_conv1 = weight_variable([5, 5, 1, 16]) #5X5 patch size , 1 input channels, 16 output channels
b_conv1 = bias_variable([16]) # bias vector for each output channel
x_image = tf.reshape(x, [-1, 28, 28, 1]) #To apply the layer, we first reshape x to a 4d tensor, 28x28 is WidthxHeight

# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
# The max_pool_2x2 method will reduce the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
# In order to build a deep network, we stack several layers of this type. 
# The second layer will have 32 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 16, 32]) #5X5 patch size , 1 input channels, 32 output channels
b_conv2 = bias_variable([32]) # bias vector for each output channel

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 128 neurons to allow processing on entire image. 

W_fc1 = weight_variable([7 * 7 * 32, 128]) #final
b_fc1 = bias_variable([128]) #final

# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout
# To reduce overfitting, we will apply dropout before the readout layer
# We create a placeholder for the probability that a neuron's output is kept during dropout. 
# This allows us to turn dropout on during training, and turn it off during testing. 

keep_prob = tf.placeholder(tf.float32)

# tf.nn.dropout op automatically handles scaling neuron outputs
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
# Finally, we add a layer, just like for the one layer softmax regression above.
W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Adam optimiser better than gradient descent
# loss/cost function, accuracy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# craeate graph
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.histogram('cross_hist', cross_entropy)
merged = tf.summary.merge_all()
trainwriter = tf.summary.FileWriter('data/logs', sess.graph)
sess.run(tf.global_variables_initializer())
# train 500 iterations
for i in range(500):
    batch = mnist.train.next_batch(50)
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    trainwriter.add_summary(summary, i)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
# test/evaluate
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
end = int(round(time.time() * 1000))
print("Time for building convnet: ")
print(end - start)
