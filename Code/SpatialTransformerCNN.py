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

def weight_vector(kernel_size):
	"""
	Generates a weight vector for corresponding kernel size

	Args:
		kernel_size: size of the weight kernel


def model(opt):









	

