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

def read_data_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('/content/gdrive/My Drive/CZ4042 Dataset/train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])        #Uses the first paragraph
      y_train.append(int(row[0]))

  with open('/content/gdrive/My Drive/CZ4042 Dataset/test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])         #Uses the first paragraph 
      y_test.append(int(row[0]))
  
  x_train = pd.Series(x_train)
  y_train = pd.Series(y_train)
  x_test = pd.Series(x_test)
  y_test = pd.Series(y_test)
  
  x_train = pd.Series(x_train)
  y_train = pd.Series(y_train)
  x_test = pd.Series(x_test)
  y_test = pd.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  num_words = len(vocab_processor.vocabulary_)
  
  return x_train, y_train, x_test, y_test, num_words

def genAccLossFig4(epoch, train_loss, train_acc, test_acc, name=''):
  plt.plot(range(len(train_loss)), train_loss, label='train loss')
  plt.plot(range(len(train_acc)), train_acc, label='train acc')
  plt.plot(range(len(test_acc)), test_acc, label='test acc')
  plt.text(len(train_loss), train_loss[-1], "%.4f" % train_loss[-1])
  plt.text(len(train_acc), train_acc[-1], "%.4f" % train_acc[-1])
  plt.text(len(test_acc), test_acc[-1], "%.4f" % test_acc[-1])
  plt.legend()
  plt.title('Word RNN - Accuracy / Loss of Model Iteration:' + str(epoch))
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy/Loss')
  plt.savefig('/content/gdrive/My Drive/CZ4042 Dataset/images/Q4/Q4_Figure1_'+str(epoch)+name+'.png')
  plt.close()

HIDDEN_SIZE = 20
EMBEDDING_SIZE = 20

def rnn_words_model(x, num_words, dropout = 0.0):

  word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=num_words, embed_dim=EMBEDDING_SIZE)
  word_list = tf.unstack(word_vectors, axis=1)

  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)

  if dropout > 0:
     cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=dropout)
     

  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  return word_list, logits

def main():
  x_train, y_train, x_test, y_test, num_words = read_data_words()

  probability = [0.2, 0.4, 0.6, 0.8]

  for prob in probability:
    tf.reset_default_graph()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    inputs, logits = rnn_words_model(x, num_words, prob)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(entropy)

    # Accuracy
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

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

        inputs_, loss_, acc_ = sess.run([inputs, entropy, accuracy], feed_dict={x: x_train, y_: y_train})
        test_acc_ = accuracy.eval(feed_dict={x: x_test, y_: y_test})

        train_loss.append(loss_)
        train_acc.append(acc_)
        test_acc.append(test_acc_)

        if epoch%100 == 0:
          print('iter: %d, train_loss: %g'%(epoch, train_loss[epoch]))

        if epoch%100 == 0:
          genAccLossFig4(epoch, train_loss, train_acc, test_acc, '_'+str(prob))
          pd.DataFrame(train_loss).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q5/Q4/Q4train_loss_'+str(epoch)+'_'+str(prob)+'.csv')
          pd.DataFrame(train_acc).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q5/Q4/Q4train_acc_'+str(epoch)+'_'+str(prob)+'.csv')
          pd.DataFrame(test_acc).to_csv('/content/gdrive/My Drive/CZ4042 Dataset/csv/Q5/Q4/Q4test_acc_'+str(epoch)+'_'+str(prob)+'.csv')
      
    print("End of dropout with %0.1f" % prob)

  print("==========END===========")

if __name__ == '__main__':
  main()