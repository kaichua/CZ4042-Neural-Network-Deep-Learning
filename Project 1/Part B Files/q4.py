import tensorflow as tf
import numpy as np
import pylab as plt
from sklearn import preprocessing
import multiprocessing as mp

from constants import *
import q3a

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 6

num_neuron = 50
seed = SEED
np.random.seed(seed)
        
def train(keep_prob, num_layer):
    # first load the data into numpy arrays
    admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
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
    x = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 1])
    keep_prob_ = tf.placeholder(tf.float32)
    # keep_prob_ = keep_prob

    # Build the graph for the deep net
    input_weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neuron], stddev=1.0 / np.sqrt(NUM_FEATURES), seed=seed), name='input_weights')
    input_biases = tf.Variable(tf.zeros([BATCH_SIZE, num_neuron]), dtype=tf.float32, name='input_biases')
    
    zOne = tf.matmul(x, input_weights) + input_biases
    hOne = tf.nn.relu(zOne, name='hidden_layer_one')
    hOne_dropout = tf.nn.dropout(hOne, keep_prob_)
    if (num_layer == 4):
        hidden_weights = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32, seed=seed), dtype=tf.float32, name='hidden_weights')
        hidden_biases = tf.Variable(tf.zeros([BATCH_SIZE, num_neuron]), dtype=tf.float32, name='hidden_biases')
        zTwo = tf.matmul(hOne_dropout, hidden_weights) + hidden_biases
        hTwo = tf.nn.relu(zTwo, name='hidden_layer_two')
        hTwo_dropout = tf.nn.dropout(hTwo, keep_prob_)

        output_weights = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), seed=seed), name='output_weights')
        output_bias = tf.Variable(tf.zeros([BATCH_SIZE, 1]), name='output_bias')
        y = tf.matmul(hTwo_dropout, output_weights) + output_bias

        # add l2 regularization on weights
        regularization = tf.nn.l2_loss(input_weights) + tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(output_weights)
    elif (num_layer == 5):
        hidden_weights_one = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32, seed=seed), dtype=tf.float32, name='hidden_weights_one')
        hidden_biases_one = tf.Variable(tf.zeros([BATCH_SIZE, num_neuron]), dtype=tf.float32, name='hidden_biases_one')
        zTwo = tf.matmul(hOne_dropout, hidden_weights_one) + hidden_biases_one
        hTwo = tf.nn.relu(zTwo, name='hidden_layer_two')
        hTwo_dropout = tf.nn.dropout(hTwo, keep_prob_)

        hidden_weights_two = tf.Variable(tf.truncated_normal([num_neuron, num_neuron], stddev=1.0 / np.sqrt(num_neuron), dtype=tf.float32, seed=seed), dtype=tf.float32, name='hidden_weights_two')
        hidden_biases_two = tf.Variable(tf.zeros([BATCH_SIZE, num_neuron]), dtype=tf.float32, name='hidden_biases_two')
        zThree = tf.matmul(hTwo_dropout, hidden_weights_two) + hidden_biases_two
        hThree = tf.nn.relu(zThree, name='hidden_layer_three')
        hThree_dropout = tf.nn.dropout(hThree, keep_prob_)

        output_weights = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=1.0 / np.sqrt(num_neuron), seed=seed), name='output_weights')
        output_bias = tf.Variable(tf.zeros([BATCH_SIZE, 1]), name='output_bias')
        y = tf.matmul(hThree_dropout, output_weights) + output_bias

        # add l2 regularization on weights
        regularization = tf.nn.l2_loss(input_weights) +\
            tf.nn.l2_loss(hidden_weights_one) +\
            tf.nn.l2_loss(hidden_weights_two) +\
            tf.nn.l2_loss(output_weights)

    #Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis = 1))
    loss_with_regularization = mse + BETA*regularization
    train_op = optimizer.minimize(loss_with_regularization)

    sess = tf.Session()
    train_err = []
    test_err = []
    with sess.as_default():
        sess.run(tf.global_variables_initializer())

        for i in range(1, EPOCHS+1):

            lastJ = 0
            cumTrainError = 0

            for j in range(1, trainX.shape[0] // 8):
                train_op.run(feed_dict={x: trainX[lastJ:j*8], 
                                        y_: trainY[lastJ:j*8],
                                        keep_prob_: keep_prob})
                cumTrainError += mse.eval(feed_dict={x: trainX[lastJ:j*8], 
                                                    y_: trainY[lastJ:j*8],
                                                    keep_prob_: keep_prob})
                lastJ = j*8
            avgTrainError = cumTrainError / (trainX.shape[0] // 8)
            train_err.append(avgTrainError)

            lastJ = 0
            cumTestError = 0
            for j in range(1, testX.shape[0] // 8):
                cumTestError += mse.eval(feed_dict={x: testX[lastJ:j*8], 
                                                    y_: testY[lastJ:j*8],
                                                    keep_prob_: 1})
                lastJ = j*8
            avgTestError = cumTestError / (testX.shape[0] // 8)
            test_err.append(avgTestError)

            if i % 100 == 0 or i == 1:
                print('num_layer: %d keep_prob: %f iter %d: train error %g test error %g'%(num_layer, keep_prob, i, train_err[i-1], test_err[i-1]))
        print('num_layer: %d keep_prob: %f iter %d: train error %g test error %g'%(num_layer, keep_prob, i, train_err[i-1], test_err[i-1]))
    sess.close()
    return test_err

def main():
    no_threads = mp.cpu_count()
    test_loss = q3a.q3a()
    p = mp.Pool(processes = no_threads)
    # test_loss = p.starmap(train, [(1.0, 3), (1.0, 4), (0.8, 4), (1.0, 5), (0.8, 5)])
    # test_loss = p.starmap(train, [(1.0, 3)])
    test_loss.append(p.starmap(train, [(1.0, 4), (0.8, 4), (1.0, 5), (0.8, 5)]))

    labels = ['Original 3 Layer Graph with 10 neurons only and no dropout', \
                '4 Layer Graph with no dropout', \
                '4 Layer Graph with 0.8 dropout', \
                '5 Layer Graph with no Dropout', \
                '5 Layer Graph with 0.8 Dropout']

    # plot learning curves
    plt.figure()
    for i in range(len(test_loss)):
        plt.plot(range(EPOCHS), test_loss[i], label=labels[i])

    plt.xlabel(str(EPOCHS) + ' iterations')
    plt.ylabel('Test Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

