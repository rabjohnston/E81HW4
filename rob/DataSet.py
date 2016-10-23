import numpy as np
from six.moves import cPickle as pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

class DataSet():

    def __init__(self):
        self.label_name = None
        self.data = None
        self.labels = None
        self.image = None

        self.test_dataset =None
        self.train_dataset = None

        self.test_labels = None
        self.train_labels = None

        self._image_size = 32
        self._num_channels = 3

    def unpickle(self, file):
        fo = open(file, 'rb')
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
        return dict

    def to_image(self, X):
        return X.reshape((X.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)

    def load(self):
        # Load data files
        b1 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b2 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b3 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b4 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b5 = self.unpickle('cifar-10-batches-py/data_batch_1')

        print(b1[b'data'])

        # Concatenate data
        self.data = np.vstack([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']]).astype(np.float32)

        # Convert to image (for visualisation purposes)
        self.image = self.to_image(self.data)

        # Reformat
        self.data = self.data.reshape((-1, self._image_size * self._image_size * self._num_channels)).astype(np.float32)

        # Concatenate labels
        labels = np.hstack([b1[b'labels'], b2[b'labels'], b3[b'labels'], b4[b'labels'], b5[b'labels']])
        print('Labels: ', labels.shape)

        # Reformat labels
        num_labels = 10
        self.labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
        print('Labels (2): ', self.labels.shape)

        # Pull out the names of the labels
        meta = self.unpickle('cifar-10-batches-py/batches.meta')
        self.label_name = meta[b'label_names']




        self.data = normalize(self.data, axis=0)

        self.train_dataset, self.test_dataset, self.train_labels, self.test_labels = train_test_split(self.data, self.labels, test_size=.33, random_state=10)

        #print(test[b'label_names'])


    def display(self, id):
        print('Display')
        import matplotlib.pyplot as plt
        plt.imshow(self.image[id])
        plt.title(self.label_name[self.labels[id]])
        plt.show()