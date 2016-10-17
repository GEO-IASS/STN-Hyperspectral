# -*- coding: utf-8 -*-

"""
author : Kaustubh Mani

In an attempt to add transformation invariance to Convolutional 
Neural Network trained on Hyperspectral Images obtained from LandSat
images

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import pandas as pd 
import numpy as np 
import scipy.io as io
from spatial_transformer import transformer 
import math
import time


# Command Line Arguments

flags = tf.app.flags

flags.DEFINE_integer('pixels', 11, 'Dimension of input patches')
flags.DEFINE_integer('kernel_dim', 3, 'Dimension of the convolution kernel')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 4000, 'Number of epochs to train.')
flags.DEFINE_integer('conv1', 500, 'Number of filters in convolutional layer 1.')
flags.DEFINE_integer('conv2', 100, 'Number of filters in convolutional layer 2.')
flags.DEFINE_integer('fc1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('fc2', 84, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Mini Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('dropout', 1.0, ' Amount of droupout for regularization')

opt = flags.FLAGS

# Defining Global Variables 

NUM_CLASS = 16 # number of output classes
CHANNELS = 220
IMAGE_PIXELS = opt.pixels * opt.pixels * CHANNELS

def load_Data(dim = 11):

	"""
	Args:
		dim: dimension of the input patches
	Returns:
		x_train,y_train,x_test,y_test: Train and test data

	"""

	x_train = np.concatenate([io.loadmat("../Data/Train_" + str(dim) + "_0_" + str(i) + ".mat")["train_patch"] for i in range(1,9)], 0)

	y_train = np.concatenate([io.loadmat("../Data/Train_" + str(dim) + "_0_"  + str(i) + ".mat")["train_labels"][0] for i in range(1,9)], 0)

	x_test = np.concatenate([io.loadmat("../Data/Test_" + str(dim) + "_90_" + str(i) + ".mat")["test_patch"] for i in range(1,7)], 0)

	y_test = np.concatenate([io.loadmat("../Data/Test_" + str(dim) + "_90_" + str(i) + ".mat")["test_labels"][0] for i in range(1,7)], 0)

	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	return x_train, y_train, x_test, y_test


def init_placeholders(batch_size):

	"""
	Defining placeholders for data flow

	Args:
		batch_size: mini-batch size
	Returns:
		x, y: placeholder for x and y

	"""

	x = tf.placeholder(tf.float32, [opt.batch_size, IMAGE_PIXELS])
	y = tf.placeholder(tf.float32, [opt.batch_size, NUM_CLASS])

	return x, y

def dense_to_one_hot(labels, n_classes=2):

    """
    Convert class labels from scalars to one-hot vectors.

	Args:
		labels: labels to be encoded
		n_classes: number of classes in the output
	Returns:
		labels_one_hot: encoded labels in OHE

    """
    labels_one_hot = []
    for label in list(labels):
    	temp = np.zeros(n_classes, dtype=np.float32)
    	temp[int(label)-1] = 1
    	labels_one_hot.append(temp)
    labels_one_hot = np.array(labels_one_hot)

    return labels_one_hot

def weight_vector(shape):
	"""
	Generates a weight vector for corresponding shape 

	Args:
		shape: shape of the weight vector
	Returns:
		weight: weight vector

	"""

	weight = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return weight

def bias_vector(shape):
	"""
	Generates bias vector for corresponding shape 

	Args:
		shape: shape of the bias vector
	Returns: 
		bias: bias vector

	"""
	bias = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return bias


def spatial_transformer(x, opt, keep_prob, out_size):
	"""
	Generates spatial transformer network by setting up the two-layer localization network
	to figure out the parameters for an affine transformation of the input

	Args:
		x: input vector
		name: name of the spatial transformer
	Returns:
		h_trans: transformed feature map (tensor)

	"""
	x_tensor = tf.reshape(x, [-1, opt.pixels, opt.pixels, CHANNELS])

	# Weights for localization network
	W_fc_loc1 = weight_vector([IMAGE_PIXELS, 20])
	b_fc_loc1 = bias_vector([20])
	W_fc_loc2 = weight_vector([20, 6])
	# starting with identity transformation
	initial = np.array([[1., 0, 0], [0, 1., 0]])
	initial = initial.astype('float32')
	initial = initial.flatten()
	b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
	# Defining two layer localization network
	h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1) 
	# h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
	h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)

	h_trans = transformer(x_tensor, h_fc_loc2, out_size)

	return h_trans


def convolution_layer(x, n_filters, opt, kernel_size=3, padding="SAME", name="conv"):
	"""
	Generates the convolution layer with the parameters specified

	Args:
		x: input vector
		n_filters: number of filters in the conv layer
		kernel_size: size of the convolution kernel
		stride: stride between two succesive convolutions
		padding: "SAME" OR "VALID"
		name: name of the convolution layer on graph
	Returns:
		conv: convolution output

	"""
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, CHANNELS, n_filters], 
		                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
		biases = tf.get_variable('biases', shape=[n_filters], initializer=tf.constant_initializer(0.05))

		x_new = tf.reshape(x, [-1,opt.pixels, opt.pixels, CHANNELS])
		conv = tf.nn.conv2d(x_new, weights, strides=[1, 1, 1, 1], padding=padding)
		conv = tf.nn.relu(conv + biases)

    	return conv

def fullyconnected_layer(x, n_units, 
						name = 'fc',
						stddev = 0.02):

	"""
	Generates the fully connected layers with the parameters specified

	Args:
		x: input_vector
		n_units: number of units to connect to 
		name: name of the fully connected layer
		stddev: standard deviation for variable initialization
	Returns:
		fc: fully connected outputs

	"""
	shape = x.get_shape().as_list()

	with tf.variable_scope(name) as scope:
		weights = tf.Variable(
		    tf.truncated_normal([shape[1], n_units],
		                        stddev=1.0 / math.sqrt(float(shape[1]))),
		    name='weights')
		biases = tf.Variable(tf.zeros([n_units]),
		                     name='biases')
		fc = tf.nn.relu(tf.matmul(x, weights) + biases, name=name)

		return fc

def softmax(x, n_units, name='softmax'):
	"""
	Softmax layer on top of the fully connected layer

	Args:
		x: input vector
		n_units: number of units to connect to 
	Returns:
		logits: logit vector containing scores for every class

	"""
	with tf.variable_scope(name) as scope:
		weights = tf.Variable(
			tf.truncated_normal([n_units, NUM_CLASS],
										stddev=1.0 / math.sqrt(float(n_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASS]),
							name='biases')

		logits = tf.matmul(x, weights) + biases

		return logits



def eval_loss(logits, labels):
	"""
	Calculates the loss from the logits and the labels.

	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].
	Returns:
		loss: Loss tensor of type float.

	"""
	labels = tf.to_float(labels)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
												logits, labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss

def optimizer_step(loss, learning_rate):
	"""
	Create summary to keep track of the loss after iterations in 
	tensorboard and optimizes the loss function using Adam Optimizer

	Args:
		loss: loss value obtained from eval_loss()
		learning_rate: learning rate for SGD
	Returns:
		optim_op: optimizer op

	"""
	# adding summary to keep track of loss
	tf.scalar_summary(loss.op.name, loss)
	# using Adam Optimizer as the optimizer function
	optimizer = tf.train.AdamOptimizer(learning_rate)
	# track the global step
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# use the optimizer to minimize the loss 
	optim_op = optimizer.minimize(loss, global_step=global_step)

	return optim_op

def evaluate(logits, labels):
	"""
	Evaluating accuracy of the network

	Args:
		logits: logits tensor
		labels: labels corresponding to a particular batch
	Returns:
		accuracy: accuracy of the network

	"""
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

	return accuracy

def evaluate_model(sess, eval_correct, x, y, x_data, y_data, opt):
	"""
	Evaluates the model 
	Args:
		sess: working session on graph
		x: placeholder for patch
		y: placeholder for label
		x_data: input patch
		y_data: output label
	Returns:

	"""
	true_count = 0  # Counts the number of correct predictions.
	steps_per_epoch = x_data // batch_size
	num_examples = steps_per_epoch * batch_size
	for i in range(steps_per_epoch + 1):

		if i == num_batch:
			batch_xs = x_train[i*opt.batch_size:]
			batch_ys = y_train[i*opt.batch_size:]
		else:
			batch_xs = x_train[i*opt.batch_size:(i+1)*opt.batch_size]
			batch_ys = y_train[i*opt.batch_size:(i+1)*opt.batch_size]

		batch_xs = batch_xs.reshape(opt.batch_size, IMAGE_PIXELS)

		true_count += sess.run(eval_correct,
		                       feed_dict={
		                    		x: batch_xs,
		                    		y: batch_ys,
		                    		keep_prob: opt.dropout
		                       })

	precision = true_count / num_examples
	print(' Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def start_trainer(opt):

	"""
	Initiate training of the Spatial Transformer CNN network

	Args:
		opt: command line arguments (dictionary)

	"""

	# Load the data set
	x_train, y_train, x_test, y_test = load_Data(opt.pixels)
	y_train = dense_to_one_hot(y_train, 16)
	y_test = dense_to_one_hot(y_test, 16)

	with tf.Graph().as_default():

		keep_prob = tf.placeholder(tf.float32)

		x, y = init_placeholders(opt.batch_size)
		# Building the graph
		# stn1
		h_trans = spatial_transformer(x, opt, keep_prob, x_train.shape[1:])
		print(h_trans)
		# conv1
		h_conv1 = convolution_layer(h_trans, opt.conv1, opt, kernel_size=opt.kernel_dim, padding="VALID", name="conv1")
		# pool1
		h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
		            			strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
		# conv2
		h_conv2 = convolution_layer(h_pool1, opt.conv2, opt, kernel_size=opt.kernel_dim, padding="VALID", name="conv2")
		# pool2
		h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
		# Size of the feature map after two convolutions and two pooling layers
		size_after_conv_and_pool_twice = int(math.ceil((math.ceil(float(opt.pixels-opt.kernel_dim+1)/2)-opt.kernel_dim+1)/2))
		# Converting pool2 feature map from 4D TO 2D and feed it to fully connected layer
		h_pool2_flat = tf.reshape(h_pool2, [-1, (size_after_conv_and_pool_twice**2)*opt.conv2])
		# fc1
		h_fc1 = fullyconnected_layer(h_pool2_flat, opt.fc1, name="fc1")
		#dropout
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		# fc2
		h_fc2 = fullyconnected_layer(h_fc1, opt.fc2, name="fc2")
		# softmax
		logits = softmax(h_fc2, opt.fc2, name='softmax')

		# loss 
		loss = eval_loss(logits, y)
		#optimize step
		optim_op = optimizer_step(loss, opt.learning_rate)
		# correct eval
		eval_correct = evaluate(logits, y)

		# initializing all variables on the graph
		init = tf.initialize_all_variables()

		# saver for recording training 
		saver = tf.train.Saver()

		# creating a session for running ops on graph
		sess = tf.Session()

		# run initializer in the defined session
		sess.run(init)

		num_batch = int(len(x_train)/opt.batch_size)

		for epoch in xrange(opt.num_epochs):
			# start time
			start = time.time()

			for i in range(num_batch + 1):

				if i == num_batch:
					batch_xs = x_train[i*opt.batch_size:]
					batch_ys = y_train[i*opt.batch_size:]
				else:
					batch_xs = x_train[i*opt.batch_size:(i+1)*opt.batch_size]
					batch_ys = y_train[i*opt.batch_size:(i+1)*opt.batch_size]

				batch_xs = batch_xs.reshape(opt.batch_size, IMAGE_PIXELS)

				_, loss_value = sess.run([optim_op, loss],
				                        feed_dict={
				                    		x: batch_xs,
				                    		y: batch_ys
				                        })

			end_time = time.time() - start

			if epoch % 50 == 0:
				print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, end_time))

			if (epoch + 1) % 1000 == 0 or (epoch + 1) == opt.num_epochs:
				saver.save(sess, 'model-STN-CNN-'+str(opt.pixels)+'X'+str(opt.pixels)+'.ckpt', global_step=step)
				# Evaluate on train data
				print("Training Data Evaluation:")
				evaluate_model(sess, eval_correct, x, y, x_train, y_train, opt)

				print(" Test Data Evaluation:")
				evaluate_model(sess, eval_correct, x, y, x_test, y_test, opt)


start_trainer(opt)

