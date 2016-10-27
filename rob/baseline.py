# -*- coding: utf-8 -*-

from dataset import DataSet
import tensorflow as tf
import numpy as np


class Baseline:
    """
    Model base. This allows us to build models and run them through tensorflow
    """
    def __init__(self, ds, name):

        # The cifar-10 datas
        self._ds = ds

        # The name of the model
        self.name = name

        # The batch size we're using
        self._batch_size = 0

        # Tensor flow variables for the training and test data
        self.tf_train_dataset = None
        self.tf_train_labels = None
        self.tf_valid_dataset = None
        self.tf_test_dataset = None

        # Tensor flow variable
        self.tf_keep_prob = None



        # The optimiser function
        self.optimizer = None

        # The loss function
        self.loss = None

        self.graph = None

        # The hyper parameters. These are stored in a dict so we can output them to file after a run.
        # This will be useful when we're evaluating the model
        self.params = None

        # After training we'll save the predictions on the training and test data
        self.train_prediction = None
        self.valid_prediction = None
        self.test_prediction = None

        # After training we'll also store the test predicitons
        self.test_preds = None

    # FLAGS = tf.app.flags.FLAGS
    #
    # # Basic model parameters.
    # tf.app.flags.DEFINE_integer('batch_size', 128,
    #                             """Number of images to process in a batch.""")
    # tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
    #                            """Path to the CIFAR-10 data directory.""")
    # tf.app.flags.DEFINE_boolean('use_fp16', False,
    #                             """Train the model using fp16.""")
    #
    # def _variable_on_cpu(name, shape, initializer):
    #     """Helper to create a Variable stored on CPU memory.
    #     Args:
    #       name: name of the variable
    #       shape: list of ints
    #       initializer: initializer for Variable
    #     Returns:
    #       Variable Tensor
    #     """
    #     with tf.device('/cpu:0'):
    #         #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    #         var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    #
    #     return var

    def accuracy(self, predictions, labels):
        """

        :param predictions:
        :param labels:
        :return:
        """
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])


    def run_session(self, num_batches, k_prob=1.0):
        """

        :param num_batches:
        :param k_prob:
        :return:
        """

        # config = tf.ConfigProto(
        #     device_count={'GPU': 0 if run_on_cpu else 1}
        # )


        with tf.Session(graph = self.graph) as session:

            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", session.graph)
            tf.initialize_all_variables().run()
            print("Initialized")
            for batch in range(num_batches):
                offset = (batch * self._batch_size) % (self._ds.train_labels.shape[0] - self._batch_size)
                batch_data = self._ds.train_dataset[offset:(offset + self._batch_size), :]
                batch_labels = self._ds.train_labels[offset:(offset + self._batch_size), :]
                feed_dict = {self.tf_train_dataset: batch_data, self.tf_train_labels: batch_labels, self.tf_keep_prob: k_prob}

                _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

                # Print out status of run
                if (batch % 500 == 0):
                    print("Minibatch loss at batch {}: {}".format(batch, l))

                    train_accuracy = self.accuracy(predictions, batch_labels)
                    print("Minibatch accuracy: {:.1f}".format(train_accuracy))

                    valid_accuracy = self.accuracy(self.valid_prediction.eval(), self._ds.valid_labels)
                    print("Validation accuracy: {:.1f}".format(valid_accuracy))

            accuracy = self.accuracy(self.test_prediction.eval(), self._ds.test_labels)
            self.params['accuracy'] = accuracy
            print("Test accuracy ({}): {:.1f}".format(self.name, accuracy))
            self.test_preds = self.test_prediction.eval().ravel()
