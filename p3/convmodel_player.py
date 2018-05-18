#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cv2

def myModel(X, d1, num_classes, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=5, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        d2 = reduce(lambda x, y: x*y, list(o4.shape.as_list()[1:]))
        h = tf.layers.dense(inputs=tf.reshape(o4, [d1, d2]), units=32, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y

x = tf.placeholder(tf.float32, [None, 80, 140, 1])
y = myModel(x, 1, 3)

# PLAY
#  cap = cv2.VideoCapture(0)
url='http://192.168.0.16:8080/shot.jpg'

saver = tf.train.Saver()
import urllib
from time import sleep

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    while True:
        #  ret, img = cap.read()  # 720x1280x3 <-- print(img.shape);
        imgResp = urllib.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        #  cv2.imshow('Capture', img)
        #  sleep(10000)

        resized = cv2.resize(img, (140, 80), interpolation=cv2.INTER_AREA)
        #  cropped = resized[0:180, 70:250]
        #  resized64 = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        gray = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))

        cv2.imshow('Capture', gray)
        frame = gray.reshape(-1, 80, 140, 1)
        print(sess.run(y, feed_dict={x: frame}))
        ch = 0xFF & cv2.waitKey(10)
        if ch == 27:
            break

cv2.destroyAllWindows()
