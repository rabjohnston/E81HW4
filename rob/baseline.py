# -*- coding: utf-8 -*-

from dataset import DataSet
import tensorflow as tf
import numpy as np


class Baseline:

    def __init__(self, ds, name):
        self._ds = ds
        self._name = name

        self._batch_size = 0
        self.tf_train_dataset = None
        self.tf_train_labels = None
        self.tf_keep_prob = None

        self.train_prediction = None
        self.test_prediction = None

        self.optimizer = None
        self.loss = None

        self.test_preds = None

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])

    def run_session(self, num_epochs, k_prob=1.0):

        with tf.Session(graph= self.graph) as session:

            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", session.graph)
            tf.initialize_all_variables().run()
            print("Initialized")
            for epoch in range(num_epochs):
                offset = (epoch * self._batch_size) % (self._ds.train_labels.shape[0] - self._batch_size)
                batch_data = self._ds.train_dataset[offset:(offset + self._batch_size), :]
                batch_labels = self._ds.train_labels[offset:(offset + self._batch_size), :]
                feed_dict = {self.tf_train_dataset: batch_data, self.tf_train_labels: batch_labels, self.tf_keep_prob: k_prob}

                _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

                if (epoch % 500 == 0):
                    print("Minibatch loss at epoch {}: {}".format(epoch, l))
                    print("Minibatch accuracy: {:.1f}".format(self.accuracy(predictions, batch_labels)))
                    #print("Validation accuracy: {:.1f}".format(self.accuracy(self.valid_prediction.eval(), self.valid_labels)))
            print("Test accuracy: {:.1f}".format(self.accuracy(self.test_prediction.eval(), self._ds.test_labels)))
            self.test_preds = self.test_prediction.eval().ravel()
