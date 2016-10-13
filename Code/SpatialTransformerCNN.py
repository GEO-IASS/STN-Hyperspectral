"""
author : Kaustubh Mani

In an attempt to add transformation invariance to Convolutional 
Neural Network trained on Hyperspectral Images obtained from LandSat
images

"""

import tensorflow as tf 
import pandas as pd 
import numpy as np 
import scipy.io as io
from spatial_transformer import transformer 


# Command Line Arguments

flags = tf.app.flags

flags.DEFINE_integer('pixels', 11, 'Dimension of input patches')
flags.DEFINE_integer('kernel_dim', 3, 'Dimension of the convolution kernel')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 4000, 'Number of steps to run trainer.')
flags.DEFINE_integer('conv1', 500, 'Number of filters in convolutional layer 1.')
flags.DEFINE_integer('conv2', 100, 'Number of filters in convolutional layer 2.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 84, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Mini Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('dropout', 0.5, ' Amount of droupout for regularization')

opt = flags.FLAGS

# Defining Global Variables 

NUM_CLASS = 16 # number of output classes
CHANNELS = 220
IMAGE_PIXELS = opt.pixels * opt.pixels * CHANNELS

def load_Data(dim = 11):

	"""
	Args:
		dim: dimension of the input patches
	Output:
		x_train,y_train,x_test,y_test: Train and test data

	"""

	x_train = tf.concat(0, [io.loadmat("../Data/Train_" + str(dim) + "_0_" + str(i) + ".mat")["train_patch"] for i in range(1,9)])

	y_train = tf.concat(0, [io.loadmat("../Data/Train_" + str(dim) + "_0_"  + str(i) + ".mat")["train_labels"] for i in range(1,9)])

	x_test = tf.concat(0, [io.loadmat("../Data/Test_" + str(dim) + "_90_" + str(i) + ".mat")["test_patch"] for i in range(1,7)])

	y_test = tf.concat(0, [io.loadmat("../Data/Test_" + str(dim) + "_90_" + str(i) + ".mat")["test_labels"] for i in range(1,7)])

	return x_train, y_train, x_test, y_test


def init_placeholders(batch_size):

	"""
	Defining placeholders for data flow

	Args:
		batch_size: mini-batch size
	Output:
		x, y: placeholder for x and y

	"""

	x = tf.placeholder(tf.float32, [opt.batch_size, IMAGE_PIXELS])
	y = tf.placeholder(tf.int32, [opt.batch_size, NUM_CLASS])

	return x, y

def dense_to_one_hot(labels, n_classes=2):

    """
    Convert class labels from scalars to one-hot vectors.

	Args:
		labels: labels to be encoded
		n_classes: number of classes in the output
	Output:
		labels_one_hot: encoded labels in OHE

    """

    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1

    return labels_one_hot

def weight_vector(shape):
	"""
	Generates a weight vector for corresponding shape 

	Args:
		shape: shape of the weight vector
	Output:
		weight: weight vector

	"""

	weight = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return weight

def bias_vector(shape):
	"""
	Generates bias vector for corresponding shape 

	Args:
		shape: shape of the bias vector
	Output: 
		bias: bias vector

	"""
	bias = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return bias


def spatial_transformer(x, name = 'fc_loc'):
	"""
	Generates spatial transformer network by setting up the two-layer localization network
	to figure out the parameters for an affine transformation of the input

	Args:
		x: input vector
		name: name of the spatial transformer
	Output:

	"""
	out_size = (opt.pixels, opt.pixels)
	x_tensor = tf.reshape(x, [-1, opt.pixels, opt.pixels, 1])

	# Weights for localization network
	W_fc_loc1 = weight_variable([IMAGE_PIXELS, 20])
	b_fc_loc1 = bias_variable([20])
	W_fc_loc2 = weight_variable([20, 6])
	# starting with identity transformation
	initial = np.array([[1., 0, 0], [0, 1., 0]])
	initial = initial.astype('float32')
	initial = initial.flatten()
	b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
	# Defining two layer localization network
	h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1) 
	keep_prob = tf.placeholder(tf.float32)
	h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
	h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

	h_trans = transformer(x, h_fc_loc2, out_size)

	return h_trans


def convolution_layer(x, n_filters,
					  kernel_size = 3,stride = 1,
					  padding = "SAME", name = 'conv'):
	"""
	Generates the convolution layer with the parameters specified

	Args:
		x: input vector
		n_filters: number of filters in the conv layer
		kernel_size: size of the convolution kernel
		stride: stride between two succesive convolutions
		padding: "SAME" OR "VALID"
		name: name of the convolution layer on graph
	Outputs:
		conv: convolution output

	"""
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[kernel_size, kernel_size, CHANNELS, n_filters], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[n_filters], initializer=tf.constant_initializer(0.05))

        x_ = tf.reshape(x, [-1,opt.pixels,opt.pixels,CHANNELS])
        conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding=padding)
        conv += biases

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
	Output:
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





def model(opt):









	

