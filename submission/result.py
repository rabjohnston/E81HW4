
import pickle
import numpy as np


class Result:
    """
    Wrapper to hold a result from a run
    """

    def __init__(self):

        self.test_preds = None

        # A dict of the hyper-parameters and their values
        self.params = None

        self.optimizer_params = None

        self.epochs = None

        self.base_filename = None


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


    def load(self, base_filename):
        """
        Load in the results from a run.
        :param base_filename: the base filename of the run, eg 'saved_NN_001', where NN is the model name and 001 is the run number
        :return: Nothing
        """

        self.base_filename = base_filename

        self.test_preds = np.load('{}.preds.npy'.format(base_filename))
        self.params = self.unpickle( '{}.params'.format(base_filename))

        try:
            self.optimizer_params = self.unpickle('{}.opt'.format(base_filename))
        except:
            # There'll be no files for non-NNs
            #print("No opt params loaded")
            self.optimizer_params = None

        try:
            self.epochs = self.unpickle('{}.epochs'.format(base_filename))
        except:
            # There'll be no files for non-NNs
            #print("No epochs loaded")
            self.epochs = None





