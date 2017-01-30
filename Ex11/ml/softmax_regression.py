import tensorflow as tf
import numpy as np

from .algorithm import Algorithm

# Using the MNIST tutorial from Tensorflow (optimized for the real MNIST dataset)
class SoftmaxRegression(Algorithm):

    def __init__(self, eta=1e-3):
        # Setup model
        self.x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.matmul(self.x, W) + b

        self.y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.train_step = tf.train.AdamOptimizer(eta).minimize(cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def one_hot_vector(self, x):
        v = [0] * 10
        v[int(x)] = 1
        return v

    def fit(self, train_data, train_labels, epochs=100):
        train_labels = [self.one_hot_vector(x) for x in train_labels]
        for _ in range(epochs):
            self.sess.run(self.train_step, feed_dict={self.x: train_data, self.y_: train_labels})

    def eval(self, test_data):
        predictions = self.y.eval(feed_dict={self.x: test_data})
        return np.argmax(predictions, axis=1)

    def calc_accuracy(self, test_data, test_labels):
        test_labels = [self.one_hot_vector(x) for x in test_labels]
        return self.sess.run(self.accuracy, feed_dict={self.x: test_data, self.y_: test_labels})
