import numpy


class SplitDataset:

    def __init__(self):
        self.a_dataset = None
        self.ab_dataset = None
        self.abc_dataset = None
        self.abcd_dataset = None

        self.a_labels = None
        self.ab_labels = None
        self.abc_labels = None
        self.abcd_labels = None


    def load(self, x, y):

        total_size = x.shape[0]
        chunk_size = total_size // 4
        indices = numpy.random.permutation(x.shape[0])

        chunk1_idx = indices[:chunk_size]
        chunk2_idx = indices[:2*chunk_size]
        chunk3_idx = indices[:3*chunk_size]
        chunk4_idx = indices[:]

        self.a_dataset = x[chunk1_idx, :]
        self.ab_dataset = x[chunk2_idx, :]
        self.abc_dataset = x[chunk3_idx, :]
        self.abcd_dataset = x[chunk4_idx, :]

        self.a_labels = y[chunk1_idx, :]
        self.ab_labels = y[chunk2_idx, :]
        self.abc_labels = y[chunk3_idx, :]
        self.abcd_labels = y[chunk4_idx, :]

        print(self.a_dataset.shape)
        print(self.ab_dataset.shape)
        print(self.abc_dataset.shape)
        print(self.abcd_dataset.shape)


class SplitDataset2:

    def __init__(self):
        self.a_dataset = None
        self.ab_dataset = None
        self.abc_dataset = None
        self.abcd_dataset = None

        self.a_labels = None
        self.ab_labels = None
        self.abc_labels = None
        self.abcd_labels = None


    def load(self, x, y):

        total_size = x.shape[0]
        chunk_size = total_size // 4
        indices = numpy.random.permutation(x.shape[0])

        chunk1_idx = indices[:chunk_size]
        chunk2_idx = indices[:2*chunk_size]
        chunk3_idx = indices[:3*chunk_size]
        chunk4_idx = indices[:]

        self.a_dataset = x[chunk1_idx, :]
        self.ab_dataset = x[chunk2_idx, :]
        self.abc_dataset = x[chunk3_idx, :]
        self.abcd_dataset = x[chunk4_idx, :]

        self.a_labels = y[chunk1_idx]
        self.ab_labels = y[chunk2_idx]
        self.abc_labels = y[chunk3_idx]
        self.abcd_labels = y[chunk4_idx]

        print(self.a_dataset.shape)
        print(self.ab_dataset.shape)
        print(self.abc_dataset.shape)
        print(self.abcd_dataset.shape)