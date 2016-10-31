import numpy as np

import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

class DataSet():
    """
    Wrapper to hold the 60,000 cifar-10 images
    """

    def __init__(self, use_valid = False):

        # Are we using a validation test data set? If so then we'll split off some data
        # for validation otherwise we'll use it all for training.
        self._use_valid = use_valid

        # The name of the labels, e.g. cat, car
        self.label_name = None

        # The image data (X)
        self.data = None

        # The image data in a form that we can display as an image (for debug purposes)
        self.image = None

        # The image data that we've split off for training
        self.train_dataset =None

        # The image data that we've split off for validation
        self.valid_dataset =None

        # The test image data
        self.test_dataset = None

        # The image labels (Y)
        self.labels = None

        # The image data that we've split off for training
        self.train_labels = None

        # The image labels that we've split off for training
        self.train_labels = None

        # The test labels
        self.test_labels = None

        # The image size (32 x 32)
        self.image_size = 32

        # The number of channels (RGB = 3)
        self.num_channels = 3

        # The number of classes
        self.num_labels = 10


    def unpickle(self, file):
        """
        Unpickle one of the cifar-10 batch files
        :param file: the filename of a batch file
        :return: a dictionary of image rows
        """
        fo = open(file, 'rb')
        dict = pickle.load(fo, encoding='bytes')
        fo.close()
        return dict


    def to_image(self, X):
        """
        Reshape the supplied data to a format that we can use to display images
        :param X: the dataset to reformat
        :return: the reformatted dataset
        """
        return X.reshape((X.shape[0], self.num_channels, self.image_size, self.image_size)).transpose(0, 2, 3, 1)

    def reformat(self, X, flatten):
        if flatten:
            X = X.reshape((-1, self.image_size * self.image_size * self.num_channels)).astype(np.float32)
        else:
            X = X.reshape((-1, self.num_channels, self.image_size, self.image_size)).astype(np.float32).transpose([0,2,3,1])

        return X


    def normalise(self, X):
        return np.array(X, dtype=float) / 255.0


    def load(self, flatten=True):
        self._loadData(flatten)
        self._loadTest(flatten)

    def _loadData(self, flatten):
        """
        Load all of the cifar-10 data.
        :param flatten: determines whether we flatten the data or reshape it as a 3D structure
        :param norm: 'l1', 'l2', or 'max', optional ('l2' by default). Used to normalise the data
        :return: Nothing
        """
        # Load data files
        b1 = self.unpickle('cifar-10-batches-py/data_batch_1')
        b2 = self.unpickle('cifar-10-batches-py/data_batch_2')
        b3 = self.unpickle('cifar-10-batches-py/data_batch_3')
        b4 = self.unpickle('cifar-10-batches-py/data_batch_4')
        b5 = self.unpickle('cifar-10-batches-py/data_batch_5')

        # Concatenate data
        self.data = np.vstack([b1[b'data'], b2[b'data'], b3[b'data'], b4[b'data'], b5[b'data']]).astype(np.float32)

        # Convert to image (for visualisation purposes)
        # self.image = self.to_image(self.data)

        # Normalise - possibly not right.
        # self.data = normalize(self.data, axis=0, norm=norm)
        # Normalise between 0 and 1
        self.data = self.normalise(self.data)

        # Reformat the data. We either flatten it (useful for NN) or reshape it into a 3D structure (for CNNs)
        self.data = self.reformat( self.data, flatten )

        # Concatenate labels
        labels = np.hstack([b1[b'labels'], b2[b'labels'], b3[b'labels'], b4[b'labels'], b5[b'labels']])

        # Reformat labels
        self.labels = (np.arange(self.num_labels) == labels[:, None]).astype(np.float32)

        # Pull out the names of the labels
        meta = self.unpickle('cifar-10-batches-py/batches.meta')
        self.label_name = meta[b'label_names']

        # Split the data into test and training data
        if self._use_valid:
            self.train_dataset, self.valid_dataset, self.train_labels, self.valid_labels = \
                train_test_split(self.data, self.labels, test_size=.25, random_state=10)
        else:
            self.train_dataset = self.data
            self.train_labels = self.labels


    def _loadTest(self, flatten):

        # Load data file
        b1 = self.unpickle('cifar-10-batches-py/test_batch')

        labels = np.hstack(b1[b'labels'])
        self.test_labels = (np.arange(self.num_labels) == labels[:, None]).astype(np.float32)

        data = b1[b'data']
        data = self.normalise(data)
        data = self.reformat(data, flatten)

        # Code for QUESITON 5) - Image distortion
        # use one of the 3 disstortions bellow for each of the runs 
        # implementation of 3 ways instead of 1 for Exploratory Points
        #data = data.astype(np.float32)
        #data = np.array([np.rot90(img) for img in data])        
        #data = np.array([np.fliplr(img) for img in data]) 
        #data = np.array([np.flipud(img) for img in data]) 

        self.test_dataset = data.astype(np.float32)


    def display(self, id):
        """
        Helper function to display an image. Useful to visualise the data
        :param id: the id in the array of images.
        :return: Nothing
        """

        # I import this here as my linux environemnt doesn't have matplotlib
        import matplotlib.pyplot as plt

        if id < 0 or id > len(self.data):
            raise RuntimeError('id of image out of range.')

        plt.imshow(self.image[id])
        plt.title(self.label_name[self.labels[id]])
        plt.show()
