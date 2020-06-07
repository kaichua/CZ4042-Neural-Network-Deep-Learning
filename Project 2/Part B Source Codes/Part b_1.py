import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time

NUM_EPOCHS = 1501
LEARNING_RATE = 0.01
MAX_DOCUMENT_LENGTH = 100
MAX_LABEL = 15
SEED = 15
BATCH_SIZE = 128
NUM_CHARS = 256

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(SEED)

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('/content/gdrive/My Drive/CZ4042 Dataset/train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])       
      y_train.append(int(row[0]))

  with open('/content/gdrive/My Drive/CZ4042 Dataset/test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])         
      y_test.append(int(row[0]))
  
  x_train = pd.Series(x_train)
  y_train = pd.Series(y_train)
  x_test = pd.Series(x_test)
  y_test = pd.Series(y_test)
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test


N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

def char_cnn_model(x, dropout=0.0):
  input_layer = tf.reshape(
      tf.one_hot(x, NUM_CHARS), [-1, MAX_DOCUMENT_LENGTH, NUM_CHARS, 1])

  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if dropout > 0:
      pool1 = tf.nn.dropout(pool1, keep_prob=dropout)
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if dropout > 0:
      pool2 = tf.nn.dropout(pool2, keep_prob=dropout)
    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
  return input_layer, logits

def genAccLossFig1(epoch, train_loss, train_acc, test_acc, name=''):
  plt.plot(range(len(train_loss)), train_loss, label='train loss')
  plt.plot(range(len(train_acc)), train_acc, label='train acc')
  plt.plot(range(len(test_acc)), test_acc, label='test acc')
  plt.text(len(train_loss), train_loss[-1], "%.4f" % train_loss[-1])
  plt.text(len(train_acc), train_acc[-1], "%.4f" % train_acc[-1])
  plt.text(len(test_acc), test_acc[-1], "%.4f" % test_acc[-1])
  plt.legend()
  plt.title('Character CNN - Accuracy / Loss of Model Iteration:' + str(epoch))
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy/Loss')
  plt.savefig('/content/gdrive/My Drive/CZ4042 Dataset/images/Q1/Q1_Figure1_'+str(epoch)+name+'.png')
  plt.close()

def main():
  tf.reset_default_graph()
  x_train, y_train, x_test, y_test = read_data_chars()

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  inputs, logits = char_cnn_model(x)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(entropy)

  # Accuracy
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  start = time.time()
  print('Start time: %f' % start)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training
    train_loss, train_acc, test_acc = [], [], []
    x_len = len(x_train)
    idx = np.arange(x_len)

    for epoch in range(NUM_EPOCHS):
      np.random.shuffle(idx)
      x_train, y_train = x_train[idx], y_train[idx]

      #Mini-batch Training
      for start, end in zip(range(0, len(x_train), BATCH_SIZE), range(BATCH_SIZE, len(x_train), BATCH_SIZE)):
        batch_data = x_train[start:end]
        batch_labels = y_train[start:end]
        sess.run(train_op, feed_dict={x:batch_data, y_:batch_labels})

      loss_, acc_ = sess.run([entropy, accuracy], feed_dict={x: x_train, y_: y_train})
      test_acc_ = accuracy.eval(feed_dict={x: x_test, y_: y_test})

      train_loss.append(loss_)
      train_acc.append(acc_)
      test_acc.append(test_acc_)

      if epoch%100 == 0:
        print('iter: %d, train_loss: %g'%(epoch, train_loss[epoch]))

      if epoch%100 == 0:
        genAccLossFig1(epoch, train_loss, train_acc, test_acc)
        pd.DataFrame(train_loss).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q1/Q1train_loss_'+str(epoch)+'.csv')
        pd.DataFrame(train_acc).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q1/Q1train_acc_'+str(epoch)+'.csv')
        pd.DataFrame(test_acc).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q1/Q1test_acc_'+str(epoch)+'.csv')
        end = time.time()
        print("Time used at %d: %f"%(epoch, end-start))

  print("END")

if __name__ == '__main__':
  main()