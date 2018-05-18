# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
from functools import reduce

def one_hot(x, n):
    return [1. * (x == i) for i in range(n)]

num_classes = 3
batch_size = 5
data_folder = 'data6/'

# DATA SOURCE

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(data_folder + p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image = tf.image.decode_jpeg(file_image)
        label = one_hot(i, num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# MODEL

def myModel(X, d1, num_classes, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o = tf.layers.max_pooling2d(inputs=o, pool_size=2, strides=2)
        o = tf.layers.conv2d(inputs=o, filters=64, kernel_size=5, activation=tf.nn.relu)
        o = tf.layers.max_pooling2d(inputs=o, pool_size=2, strides=2)
        #  o5 = tf.layers.conv2d(inputs=o4, filters=64, kernel_size=5, activation=tf.nn.relu)
        #  o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)

        d2 = reduce(lambda x, y: x*y, list(o.shape.as_list()[1:]))
        o = tf.layers.dense(inputs=tf.reshape(o, [d1, d2]), units=10, activation=tf.nn.relu)
        d2 = reduce(lambda x, y: x*y, list(o.shape.as_list()[1:]))
        o = tf.layers.dense(inputs=tf.reshape(o, [d1, d2]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=o, units=num_classes, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train  = dataSource([str(i) + "/train/*.jpg" for i in range(num_classes)], batch_size=batch_size)
example_batch_valid, label_batch_valid  = dataSource([str(i) + "/valid/*.jpg" for i in range(num_classes)], batch_size=batch_size)
example_batch_test, label_batch_test    = dataSource([str(i) + "/test/*.jpg" for i in range(num_classes)],  batch_size=batch_size)

example_batch_train_predicted   = myModel(example_batch_train,  batch_size * 3, num_classes)
example_batch_valid_predicted   = myModel(example_batch_valid,  batch_size * 3, num_classes, reuse=True)
example_batch_test_predicted    = myModel(example_batch_test,   batch_size * 3, num_classes, reuse=True)


#  cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
#  cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
#  cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
#  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

cost_pattern    = lambda a, b:  tf.reduce_sum(tf.square(a - b))
cost_pattern    = lambda a, b:  tf.reduce_mean(-tf.reduce_sum(a * tf.log(b), reduction_indices=[1]))
cost_train      = cost_pattern(label_batch_train, example_batch_train_predicted)
cost_valid      = cost_pattern(label_batch_valid, example_batch_valid_predicted)

error_func                  = lambda a, b: tf.metrics.accuracy(tf.argmax(a, 1), tf.argmax(b, 1))
acc_train, update_train_op  = error_func(label_batch_train, example_batch_train_predicted)
acc_valid, update_valid_op  = error_func(label_batch_valid, example_batch_valid_predicted)
acc_test, update_test_op    = error_func(label_batch_test,  example_batch_test_predicted)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(cost_train)

# TRAINING

from matplotlib.pyplot import show, plot, legend, ylabel, subplot
saver = tf.train.Saver()
train_errors, valid_errors, test_errors = [], [], []
train_accs, valid_accs = [], []

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tab= []
    for _ in range(400):
        sess.run(optimizer)
        if _ % 10 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))

            print("Train")
            err = sess.run(cost_train)
            train_errors.append(err)
            print("Error: %.2f " % err)
            acc = sess.run(update_train_op)
            train_accs.append(acc)
            print("Accurracy: %.2f " % acc)
            print("")

            print("Valid")
            err = sess.run(cost_valid)
            valid_errors.append(err)
            print("Error: %.2f " % err)
            acc = sess.run(update_valid_op)
            valid_accs.append(acc)
            print("Accurracy: %.2f " % acc)
            print("")
            if (len(valid_accs) > 10 and max(valid_accs[5:] or [0]) > (valid_accs[-1] + .15)):
                break

    acc = sess.run(update_test_op)
    print("Test")
    print("Accuracy %.2f" % (acc * 100.))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    coord.request_stop()
    coord.join(threads)

    subplot(2, 1, 1)
    ylabel("Error")
    plot(train_errors)
    plot(valid_errors)
    legend(['Training error', 'Validation error'])

    subplot(2, 1, 2)
    ylabel("Accuracy")
    plot(train_accs)
    plot(valid_accs)
    plot([acc] * len(valid_accs))
    legend(['Training accuracy', 'Validation accuracy', 'Test accuracy'])
    show()
