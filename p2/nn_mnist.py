import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10);


x_ = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

NEURONS = 100

W1 = tf.Variable(np.float32(np.random.rand(784, NEURONS)) * .1)
b1 = tf.Variable(np.float32(np.random.rand(NEURONS)) * .1)

W2 = tf.Variable(np.float32(np.random.rand(NEURONS, 10)) * .1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * .1)

out1 = tf.nn.sigmoid(tf.matmul(x_, W1) + b1)
out2 = tf.nn.softmax(tf.matmul(out1, W2) + b2)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(out2), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


batch_s = 20
train_errors, valid_errors, train_accs, valid_accs = [],[],[],[]
acc_f = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out2, 1), tf.argmax(y_, 1)), tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in xrange(100):
    for jj in xrange(len(train_x)/batch_s):
        batch_xs = train_x[jj * batch_s : (jj + 1) * batch_s]
        batch_ys = train_y[jj * batch_s : (jj + 1) * batch_s]
        sess.run(train, feed_dict={x_: batch_xs, y_: batch_ys})

    train_errors.append(sess.run(loss, feed_dict={x_: batch_xs, y_: batch_ys}))
    valid_errors.append(sess.run(loss, feed_dict={x_: valid_x, y_: valid_y}))
    train_accs.append(sess.run(acc_f, feed_dict={x_: batch_xs, y_: batch_ys}))
    valid_accs.append(sess.run(acc_f, feed_dict={x_: valid_x, y_: valid_y}))

    if epoch >= 10 and len(set([int(x * 1000) for x in valid_accs[-5:]])) == 1:
        print('break1')
        break
    if epoch > 2 and valid_errors[-2] < valid_errors[-1]:
        print('break2')
        break
    print("--------------- %i --------------------" % epoch)
    print("training error {:.2f}, valid error {:.2f}".format(train_errors[-1], valid_errors[-1]))
    print("training acc {:.2f}, valid acc {:.2f}".format(train_accs[-1], valid_accs[-1]))

print("--------------------------------------")
print("Testing accuracy: %.2f" % (sess.run(acc_f, feed_dict={x_: test_x, y_: test_y})*100))

from matplotlib.pyplot import plot, legend, show, title, subplot

subplot(2, 1, 1)
title('Error')
p1 = plot(train_errors)
p2 = plot(valid_errors)
legend(labels=['training', 'validation'])

subplot(2, 1, 2)
title('Accuracy')
p3 = plot(train_accs)
p4 = plot(valid_accs)
legend(labels=['training', 'validation'])
show()
