#
# Project 1, starter code part b
#

import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import preprocessing

from constants import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 6

learning_rate = LEARNING_RATE
epochs = EPOCHS
batch_size = BATCH_SIZE
num_neuron = 10
seed = SEED
np.random.seed(seed)
beta = 0.001

def q3a():
	# first load the data into numpy arrays
	admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
	# for this question, drop the research component of data.
	X_data, Y_data = admit_data[1:,1:7], admit_data[1:,-1]
	Y_data = Y_data.reshape(Y_data.shape[0], 1)

	# then, shuffle the data
	idx = np.arange(X_data.shape[0])
	np.random.shuffle(idx)
	X_data, Y_data = X_data[idx], Y_data[idx]
	# then, split the data into 70:30 ratio
	splitThreshold = int(X_data.shape[0] * 70 / 100)
	trainX = X_data[:splitThreshold]
	trainY = Y_data[:splitThreshold]
	testX = X_data[splitThreshold:]
	testY = Y_data[splitThreshold:]

	# also, normalize each dataset's inputs
	scaler = preprocessing.StandardScaler()
	trainX = scaler.fit_transform(trainX)
	testX = scaler.fit_transform(testX)

	# Create the model
	x = tf.placeholder(tf.float32, [batch_size, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [batch_size, 1])

	# Build the graph for the deep net
	input_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), dtype=tf.float32, seed=seed), name='input_weights')
	input_biases = tf.Variable(tf.zeros([batch_size, num_neuron]), dtype=tf.float32, name='input_biases')
	z = tf.matmul(x, input_weights) + input_biases
	h = tf.nn.relu(z, name='hidden_layer')
	output_weights = tf.Variable(tf.truncated_normal([num_neuron, 1], seed=seed), name='output_weights')
	output_bias = tf.Variable(tf.zeros([batch_size, 1]), name='output_bias')
	y = tf.matmul(h, output_weights) + output_bias

	# add l2 regularization on weights
	regularization = tf.nn.l2_loss(input_weights) + tf.nn.l2_loss(output_weights)

	#Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	mse = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))
	loss = mse + beta*regularization
	train_op = optimizer.minimize(loss)

	sess = tf.Session()
	with sess.as_default():
		sess.run(tf.global_variables_initializer())
		train_err = []
		test_err = []
		for i in range(1, epochs+1):

			lastJ = 0
			cumTrainError = 0

			for j in range(1, trainX.shape[0] // 8):
				train_op.run(feed_dict={x: trainX[lastJ:j*8], y_: trainY[lastJ:j*8]})
				cumTrainError += mse.eval(feed_dict={x: trainX[lastJ:j*8], y_: trainY[lastJ:j*8]})
				lastJ = j*8
			avgTrainError = cumTrainError / (trainX.shape[0] // 8)
			train_err.append(avgTrainError)

			lastJ = 0
			cumTestError = 0
			for j in range(1, testX.shape[0] // 8):
				cumTestError += mse.eval(feed_dict={x: testX[lastJ:j*8], y_: testY[lastJ:j*8]})
				lastJ = j*8
			avgTestError = cumTestError / (testX.shape[0] // 8)
			test_err.append(avgTestError)

			if i % 100 == 0 or i == 1:
				print('%g,%g'%(train_err[i-1], test_err[i-1]))	
		print('%g, %g'%(train_err[i-1], test_err[i-1]))
	return test_err

if __name__ == "__main__":
	q3a()