#
# Project 2, starter code Part a
#

import tensorflow as tf # behaviour using gpu differs from behaviour using cpu
import numpy as np
import pylab as plt

import pickle
import os
import glob
import multiprocessing as mp

qn_to_attempt = 3 # change this according to the question to generate answers for

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 2000
batch_size = 128

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# types of optimizers
GD = 0
MOMENTUM = 1
RMSPROP = 2
ADAM = 3

def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_

def build_graph(images, num_filters_c1=50, num_filters_c2=60, dropout=None):

    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    #Conv 1
    # use the truncated normal weights initialization method
    W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, num_filters_c1], stddev=1.0/np.sqrt(NUM_CHANNELS*9*9)), name='weights_1')
    b1 = tf.Variable(tf.zeros([num_filters_c1]), name='biases_1')

    conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + b1)
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_1')

    # Conv_2
    W2 = tf.Variable(tf.truncated_normal([5, 5, num_filters_c1, num_filters_c2], stddev=1.0/np.sqrt(num_filters_c1*5*5)), name='weights_2')
    b2 = tf.Variable(tf.zeros([num_filters_c2]), name='biases_2')

    conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_2')

    # reshape
    dim = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim])

    # fully-connected layer
    W3 = tf.Variable(tf.truncated_normal([dim, 300], stddev=1.0/np.sqrt(dim)), name='weights_3')
    b3 = tf.Variable(tf.zeros([300]), name='biases_3')
    fc = tf.nn.relu(tf.matmul(pool_2_flat, W3) + b3)

    # dropout
    if (dropout != None):
        fc_drop = tf.nn.dropout(fc, dropout)

    #Softmax
    W4 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0/np.sqrt(300)), name='weights_4')
    b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')

    if (dropout != None):
        softmax_u = tf.matmul(fc_drop, W4) + b4
    else:
        softmax_u = tf.matmul(fc, W4) + b4

    if (dropout != None):
        return conv_1, pool_1, conv_2, pool_2, softmax_u, dropout
    else:
        return conv_1, pool_1, conv_2, pool_2, softmax_u

def main(mapping_feature_maps=True, plotting_train_test_graphs=True, num_filters_c1=50, num_filters_c2=60, optimizer=GD, dropout=1):

    # load and normalize the datasets to ranges 0 - 1
    train_images, train_labels = load_data('data_batch_1')
    train_images = (train_images - np.min(train_images, axis=0)) / np.max(train_images, axis=0)
    train_images = train_images.astype(np.float32)

    test_images, test_labels = load_data('test_batch_trim')
    test_images = (test_images - np.min(test_images, axis=0)) / np.max(test_images, axis=0)
    test_images = test_images.astype(np.float32)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.int32, [None, NUM_CLASSES])

    if (dropout < 1):
        keep_prob = tf.placeholder(tf.float32)

    if (dropout < 1):
        conv_1, pool_1, conv_2, pool_2, softmax_u, keep_prob = build_graph(x, num_filters_c1, num_filters_c2, keep_prob)
    else:
        conv_1, pool_1, conv_2, pool_2, softmax_u = build_graph(x, num_filters_c1, num_filters_c2)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=softmax_u)
    loss = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(softmax_u, 1, output_type=tf.int32),\
                                  tf.argmax(y_, 1, output_type=tf.int32))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    if (optimizer==GD):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif (optimizer==MOMENTUM):
        train_step = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    elif (optimizer==RMSPROP):
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif (optimizer==ADAM):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    N = len(train_images)
    N2 = len(test_images)
    idx = np.arange(N)
    idx2 = np.arange(N2)
    train_losses = []
    test_losses = []
    test_accuracies = []

    # enable multiprocessing on GPU
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True

    with tf.Session(config=core_config) as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            np.random.shuffle(idx)
            train_images_temp, train_labels_temp = train_images[idx], train_labels[idx]
            train_loss_ = []
            for i in range(N // batch_size):
                if (dropout < 1):
                    _, temp_loss = sess.run([train_step, loss], {x: train_images_temp[i:i + batch_size - 1],
                                                                 y_: train_labels_temp[i:i + batch_size - 1],
                                                                 keep_prob: dropout})
                else:
                    _, temp_loss = sess.run([train_step, loss], {x: train_images_temp[i:i+batch_size-1],
                                                                 y_: train_labels_temp[i:i+batch_size-1]})
                train_loss_.append(temp_loss)
            train_loss_ = np.array(train_loss_)
            train_loss_ = train_loss_.mean()
            train_losses.append(train_loss_)

            np.random.shuffle(idx2)
            test_images_temp, test_labels_temp = test_images[idx2], test_labels[idx2]

            if (dropout < 1):
                test_loss_ = loss.eval({x: test_images_temp,
                                        y_: test_labels_temp,
                                        keep_prob: 1}) # during testing, dropout must be disabled
            else:
                test_loss_ = loss.eval({x: test_images_temp,
                                   y_: test_labels_temp})
            test_losses.append(test_loss_)

            if (dropout < 1):
                test_accuracy_ = accuracy.eval({x: test_images_temp,\
                                                y_: test_labels_temp,\
                                                keep_prob: 1})
            else:
                test_accuracy_ = accuracy.eval({x: test_images_temp,\
                                                y_: test_labels_temp})
            test_accuracies.append(test_accuracy_)

            debug_text = 'epoch {} train_entropy {} test_entropy {} test_accuracy {}'\
                .format(e, train_loss_, test_loss_, test_accuracy_)
            with open('train_logs/c1Filter({})_c2Filter({})_optimizer({})_dropout({}).txt'\
                              .format(num_filters_c1, num_filters_c2, optimizer, dropout), 'a+') as f:
                f.write(debug_text + '\n')
            print(debug_text)

        # # pickle the results
        # try:
        #     with open('pickle/train_loss_c1Filter({})_c2Filters({})_optimizer({})_dropout({})'\
        #                       .format(num_filters_c1, num_filters_c2, optimizer, dropout), 'wb') as f:
        #         pickle.dump(train_losses, f)
        # except:
        #     print("error with dumping train losses: ")
        #     print(train_losses)
        # try:
        #     with open('pickle/test_accuracy_c1Filter({})_c2Filters({})_optimizer({})_dropout({})'\
        #                       .format(num_filters_c1, num_filters_c2, optimizer, dropout), 'wb') as f:
        #         pickle.dump(test_accuracies, f)
        # except:
        #     print("error with dumping test accuracies: ")
        #     print(test_accuracies)

        if (plotting_train_test_graphs):
            fig, ax = plt.subplots()
            ax.set_xlabel('epochs')
            ax.set_ylabel('training cost')
            ax.plot(np.arange(epochs), train_losses, label='original_train_loss')
            ax.set_title('training cost against epochs')

            fig2, ax2 = plt.subplots()
            ax2.set_xlabel('epochs')
            ax2.set_ylabel('testing accuracy')
            ax2.plot(np.arange(epochs), test_accuracies, label='original_test_accuracy')
            ax2.set_title('testing accuracy against epochs')
        else:
            return num_filters_c1, num_filters_c2, train_losses, test_accuracies

            # plt.legend(loc='lower right')

        # once done with training, plot the feature maps at both convolution layers and pooling layers, tgt with test patterns
        if (mapping_feature_maps):
            for i in range(2):
                current_image = test_images[i]
                current_image_to_show = current_image.copy()
                current_image_to_show = current_image_to_show.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE).transpose(1, 2, 0)
                current_image = np.reshape(current_image, [1, *current_image.shape])
                current_label = test_labels[i]
                current_label = np.reshape(current_label, [1, *current_label.shape])

                conv_1_feature_map, pool_1_feature_map, conv_2_feature_map, pool_2_feature_map  \
                    = sess.run([conv_1, pool_1, conv_2, pool_2],
                                                    {x: current_image,
                                                     y_: current_label})
                fig, main_ax_1 = plt.subplots()
                main_ax_1.imshow(current_image_to_show)
                main_ax_1.set_title('original image for sample {}'.format(i))

                fig2, main_ax_2 = plt.subplots()
                fig2.set_figheight(conv_1_feature_map.shape[1])
                fig2.set_figwidth(conv_1_feature_map.shape[2])
                main_ax_2.set_title('conv_1 feature map for sample {}'.format(i))
                plt.gray()
                for j in range(50):
                    try:
                        temp_ax = fig2.add_subplot(5, 10, j)
                        temp_ax.axis('off')
                        temp_ax.imshow(conv_1_feature_map[0, :, :, j])
                    except:
                        pass

                fig3, main_ax_3 = plt.subplots()
                fig3.set_figheight(pool_1_feature_map.shape[1])
                fig3.set_figwidth(pool_1_feature_map.shape[2])
                main_ax_3.set_title('pool_1 feature map for sample {}'.format(i))
                plt.gray()
                for j in range(50):
                    try:
                        temp_ax = fig3.add_subplot(5, 10, j)
                        temp_ax.axis('off')
                        temp_ax.imshow(pool_1_feature_map[0, :, :, j])
                    except:
                        pass

                fig4, main_ax_4 = plt.subplots()
                fig4.set_figheight(conv_2_feature_map.shape[1])
                fig4.set_figwidth(conv_2_feature_map.shape[2])
                main_ax_4.set_title('conv_2 feature map for sample {}'.format(i))
                plt.gray()
                for j in range(60):
                    try:
                        temp_ax = fig4.add_subplot(6, 10, j)
                        temp_ax.axis('off')
                        temp_ax.imshow(conv_2_feature_map[0, :, :, j])
                    except:
                        pass

                fig5, main_ax_5 = plt.subplots()
                fig5.set_figheight(pool_2_feature_map.shape[1])
                fig5.set_figwidth(pool_2_feature_map.shape[2])
                main_ax_5.set_title('pool_2 feature map for sample {}'.format(i))
                plt.gray()
                for j in range(60):
                    try:
                        temp_ax = fig5.add_subplot(6, 10, j)
                        temp_ax.axis('off')
                        temp_ax.imshow(pool_2_feature_map[0, :, :, j])
                    except:
                        pass

    plt.show()

if __name__ == '__main__':
    # clear all generated log files from directory first
    log_files = glob.glob("train_logs/*.txt")
    print(log_files)
    for f in log_files:
        os.remove(f)

    # clear all generated pickle files from directory too
    pickle_files = glob.glob("pickle/*")
    print(pickle_files)
    for f in pickle_files:
        os.remove(f)

    if (qn_to_attempt == 1):
        main() # for q1
    elif (qn_to_attempt == 2):
        # for q2
        no_threads = mp.cpu_count()
        p = mp.Pool(processes=2)

        # num_filters_c1_to_try = [16, 24, 30, 36, 42, 48, 54, 60, 64, 70, 76, 82]
        # num_filters_c2_to_try = [16, 24, 30, 36, 42, 48, 54, 60, 64, 70, 76, 82]

        # first find the region it will be in
        num_filters_c1_to_try = [16, 52, 88]
        num_filters_c2_to_try = [16, 52, 88]

        # then reduce the step size to find the exact number of filters around that region
        num_filters_c1_to_try = [70, 88, 106]
        num_filters_c2_to_try = [70, 88, 106]

        # then realize that the extreme ends could be too small
        num_filters_c1_to_try = [124, 178, 214, 250]
        num_filters_c2_to_try = [124, 178, 214, 250]


        arg_combinations = []
        #for n in num_filters_c1_to_try:
        #    for m in num_filters_c2_to_try:
		#		if (n == m and m == 88):
		#			continue
        #        arg_combinations.append((False, False, n, m))
        filter_pairs = list(zip(num_filters_c1_to_try, num_filters_c2_to_try))

        for combi in filter_pairs:
            arg_combinations.append((False, False, *combi))

        print(arg_combinations)

        all_train_losses = {}
        all_test_accuracies = {}
        for res in p.starmap(main, arg_combinations):
            num_filters_c1_used, num_filters_c2_used, train_losses_, test_accuracies_ = res
            all_train_losses['{}_{}'.format(num_filters_c1_used, num_filters_c2_used)] = train_losses_
            all_test_accuracies['{}_{}'.format(num_filters_c1_used, num_filters_c2_used)] = test_accuracies_
        # store these variables just in case
        with open('pickle/all_train_losses', 'wb') as f:
            pickle.dump(all_train_losses, f)
        with open('pickle/test_accuracies', 'wb') as f:
            pickle.dump(all_test_accuracies, f)
    elif (qn_to_attempt == 3):
        # for q3
        IDEAL_NUM_FILTERS_C1 = 214
        IDEAL_NUM_FILTERS_C2 = 214
        DROPOUT = 0.5
        no_threads = mp.cpu_count()
        print("number of threads: {}".format(no_threads))
        p = mp.Pool(processes=2)
        optimizers = [MOMENTUM, RMSPROP, ADAM]
        arg_combinations = []
        for optimizer in optimizers:
            arg_combinations.append((False, False, IDEAL_NUM_FILTERS_C1, IDEAL_NUM_FILTERS_C2, optimizer, 1))
        arg_combinations.append((False, False, IDEAL_NUM_FILTERS_C1, IDEAL_NUM_FILTERS_C2, ADAM, DROPOUT))
        for res in p.starmap(main, arg_combinations):
            print(res)
