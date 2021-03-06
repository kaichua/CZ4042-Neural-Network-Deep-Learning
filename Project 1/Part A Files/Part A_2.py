#Import the required libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os
if not os.path.isdir('proj_figures'):
    os.makedirs('proj_figures')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Given parameters 
NUM_FEATURES = 21
NUM_CLASSES = 3
batch_size = 32
k_fold = 5
hidden1_num = 10 #Using ReLU
lr = 0.01          #Learning Rate
beta = 10 ** -6    #Weight Decay Parameter Beta of 10^-6
no_epochs = 5001
snapshot = [1000, 2000, 3000, 4000, 5000]


#Normalize the training set to range min and max of 0 and 1
def normalize(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

#Read in the data required for training
def readData(csv):
    df = pd.read_csv(csv)
    X = df[df.columns[0:21]].values
    Y = df[df.columns[-1]]
    return X, Y

def shuffleSplit(X, Y, test_size):
    X, Y = shuffle(X, Y, random_state=1)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = test_size, random_state=2)
    return train_X, test_X, train_Y, test_Y
    
#Define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels-1] = 1
    return one_hot_encode

def ffn3(x , hidden_units):
    #Hidden Layer 1 with num_hidden neurons
    w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_units], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
    b1 = tf.Variable(tf.zeros([hidden_units]), name='biases')
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    #Output Layer
    w2 = tf.Variable(tf.truncated_normal([hidden_units, NUM_CLASSES], stddev=1.0/math.sqrt(float(hidden_units))), name='weights')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    u = tf.matmul(h1, w2) + b2

    l2 = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) 

    return u, l2

def genQ2FigVal(epochs, size, train_acc, test_acc, fold):
    plt.plot(range(epochs+1), train_acc, label="training acccuracy")
    plt.plot(range(epochs+1), test_acc, label="validation accuracy")
    plt.text(epochs, train_acc[-1]+0.05 , "%.3f" % train_acc[-1])
    plt.text(epochs, test_acc[-1]-0.05, "%.3f" % test_acc[-1])
    plt.title('Cross-Validation Accuracy of batch size ' + str(size) + ' for fold ' + str(fold))
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./proj_figures/PartA_2a_batch"+ str(size) + "_fold" + str(fold) + "_" + str(epochs) + "_iters.png")
    plt.close()

def genQ2aFigVal(size, model_val, model_acc, model_loss):
    plt.figure(1)
    plt.plot(range(len(size)), model_val, marker = 'x', label="accuracy", linestyle='none')
    plt.title('Cross-Validation Accuracy Among Batch Sizes')
    plt.xticks(range(len(size)), size)
    plt.xlabel('Batch sizes')
    plt.ylabel('Cross-Validation Accuracy')
    plt.savefig("./proj_figures/PartA_2a_2_val.png")
    plt.close()
    
    plt.figure(2)
    plt.plot(range(len(size)), model_loss, marker = 'x', label="accuracy", linestyle='none')
    plt.xticks(range(len(size)), size)
    plt.title('Loss Among Batch Sizes')
    plt.xlabel('Batch sizes')
    plt.ylabel('Loss')
    plt.savefig("./proj_figures/PartA_2a_2_loss.png")
    plt.close()
    
    plt.figure(3)
    plt.plot(range(len(size)), model_acc, marker = 'x', label="accuracy", linestyle='none')
    plt.xticks(range(len(size)), size)
    plt.title('Training Accuracy Among Batch Sizes')
    plt.xlabel('Batch sizes')
    plt.ylabel('Training Accuracy')
    plt.savefig("./proj_figures/PartA_2a_2_acc.png")
    plt.close()

X, Y = readData("ctg_data_cleaned.csv")
X = normalize(X, np.min(X, axis=0), np.max(X, axis=0))
Y = one_hot_encode(Y)
train_X, test_X, train_Y, test_Y = shuffleSplit(X, Y, 0.3)
x  = tf.placeholder(tf.float32, [None, NUM_FEATURES])
d = tf.placeholder(tf.float32, [None, NUM_CLASSES])

y, l2 = ffn3(x, hidden1_num)

#Cost function with L2 regularization
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=d, logits=y))
loss = tf.reduce_mean(cross_entropy + beta * l2)

#Batch GD learning
train = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

#Predictions
correct_prediction = tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(d,1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

#Begin the training
x_train_num = train_X.shape[0]

batch_trial = [4, 8, 16, 32, 64]
k_fold_data = train_X.shape[0] / k_fold
batch_acc, batch_loss, batch_validation = [], [], []

#Cross-Validation Accuracies for different batch sizes
for size in batch_trial:
    fold_acc, fold_loss, fold_validation = [], [], []
    model_acc, model_loss, model_validation = 0, 0, 0
    #Defines n fold cross validation to get the cross-validation error for this model
    print("=============Initialized (Batch size = %d) =============" % size)
    for fold in range(k_fold):
        start, end = int(fold * k_fold_data), int((fold + 1) * k_fold_data)
        x_test, y_test = train_X[start:end], train_Y[start:end]
        x_train = np.append(train_X[:start], train_X[end:], axis=0)
        y_train = np.append(train_Y[:start], train_Y[end:], axis=0)
        print("===========Running (Fold = %d) ======================" % fold)
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            train_acc, train_loss, test_acc = [], [], []
            for step in range(no_epochs):
                epoch_acc, epoch_loss, count = 0, 0, 0
                for start, end in zip(range(0, len(x_train), size), range(size, len(x_train), size)):
                    #Generate a minibatch
                    batch_data = x_train[start:end]
                    batch_labels = y_train[start:end]
                    loss_, train_ = session.run([loss, train], feed_dict={x:batch_data, d:batch_labels})
                    epoch_acc += accuracy.eval(feed_dict={x:batch_data, d:batch_labels})
                    epoch_loss += loss_ 
                    count += 1

                train_acc.append(epoch_acc / count)
                train_loss.append(epoch_loss / count)

                epoch_test_acc = accuracy.eval(feed_dict={x:x_test, d:y_test})
                test_acc.append(epoch_test_acc)
                
                if step in snapshot:
                    genQ2FigVal(step, size, train_acc, test_acc, fold)
                
                if(step % 100 == 0):
                    print('iter %d: accuracy %g loss: %g test acc: %g'%(step, train_acc[step], train_loss[step], test_acc[step]))
            
            
            fold_acc.append(train_acc[-1])
            fold_loss.append(train_loss[-1])
            fold_validation.append(test_acc[-1])
            
            print('fold %d: accuracy %g loss: %g test acc: %g'%(fold, fold_acc[fold], fold_loss[fold], fold_validation[fold]))
        
    
    for val1, val2, val3 in zip(fold_acc, fold_loss, fold_validation):
        model_acc += val1
        model_loss += val2
        model_validation += val3
    
    batch_acc.append(model_acc/k_fold)
    batch_loss.append(model_loss/k_fold)
    batch_validation.append(model_validation/k_fold)
    
    
genQ2aFigVal(batch_trial, batch_validation, batch_acc, batch_loss)
    
#Plot the time for one epoch against different batch sizes
'''
import time

no_epochs = 1
batch_time = []

for size in batch_trial:
    start_time = time.time()
    #Defines n fold cross validation to get the cross-validation error for this model
    print("=============Initialized (Batch size = %d) =============" % size)
    for fold in range(k_fold):
        start, end = int(fold * k_fold_data), int((fold + 1) * k_fold_data)
        x_test, y_test = train_X[start:end], train_Y[start:end]
        x_train = np.append(train_X[:start], train_X[end:], axis=0)
        y_train = np.append(train_Y[:start], train_Y[end:], axis=0)
        print("===========Running (Fold = %d) ======================" % fold)
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for step in range(no_epochs):
                epoch_acc, epoch_loss, count = 0, 0, 0
                for start, end in zip(range(0, len(x_train), size), range(size, len(x_train), size)):
                    #Generate a minibatch
                    batch_data = x_train[start:end]
                    batch_labels = y_train[start:end]
                    loss_, train_ = session.run([loss, train], feed_dict={x:batch_data, d:batch_labels})
                    epoch_acc += accuracy.eval(feed_dict={x:batch_data, d:batch_labels})
                    epoch_loss += loss_ 
                    count += 1
        
    end_time = time.time()
    batch_time.append(end_time - start_time)
    
plt.figure(1)
plt.plot(range(len(batch_trial)), batch_time, marker = 'x', label="time", linestyle='none')
plt.title('Time taken to train for one epoch among batch sizes')
plt.text(0.1, batch_time[0] , "%.3f" % batch_time[0])
plt.text(1.1, batch_time[1] , "%.3f" % batch_time[1])
plt.text(2.1, batch_time[2] , "%.3f" % batch_time[2])
plt.text(3.1, batch_time[3] , "%.3f" % batch_time[3])
plt.text(4.1, batch_time[4] , "%.3f" % batch_time[4])
plt.xticks(range(len(batch_trial)), batch_trial)
plt.xlabel('Batch sizes')
plt.ylabel('Time taken (seconds)')
plt.savefig("./proj_figures/PartA_2a_time.png")
plt.close()    
    
'''

#Plot the train and test accuracies for the optimal batch size
'''
batch_size = 8

def genFigAcc(epochs, train_acc, test_acc):
    plt.plot(range(epochs+1), train_acc, label="training acccuracy")
    plt.plot(range(epochs+1), test_acc, label="test accuracy")
    plt.text(epochs, train_acc[-1]+0.05 , "%.3f" % train_acc[-1])
    plt.text(epochs, test_acc[-1]-0.05, "%.3f" % test_acc[-1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig("./proj_figures/PartA_2c_1_" + str(epochs) + "_iters.png")
    plt.close()


def genFigLoss(epochs, train_loss):
    plt.plot(range(epochs+1), train_loss, label="training loss")
    plt.text(epochs, train_loss[-1], "%.4f" % train_loss[-1])
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./proj_figures/PartA_2c_2_" + str(epochs) + "_iters.png")
    plt.close()

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print("=============Initialized=============")
    train_acc, train_loss, test_acc = [], [], []
    for step in range(no_epochs):
        epoch_acc, epoch_loss, count = 0, 0, 0
        for start, end in zip(range(0, x_train_num, batch_size), range(batch_size, x_train_num, batch_size)):
            #Generate a minibatch
            batch_data = train_X[start:end]
            batch_labels = train_Y[start:end]
            loss_, train_ = session.run([loss, train], feed_dict={x:batch_data, d:batch_labels})
            epoch_acc += accuracy.eval(feed_dict={x:batch_data, d:batch_labels})
            epoch_loss += loss_ 
            count += 1
        
        train_acc.append(epoch_acc / count)
        train_loss.append(epoch_loss / count)
                  
        epoch_test_acc = accuracy.eval(feed_dict={x:test_X, d:test_Y})
        test_acc.append(epoch_test_acc)
        
        if(step % 100 == 0):
            print('iter %d: accuracy %g loss: %g test acc: %g'%(step, train_acc[step], train_loss[step], test_acc[step]))

        if step in snapshot:
            genFigAcc(step, train_acc, test_acc)
            genFigLoss(step, train_loss)
'''