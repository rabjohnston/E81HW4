import numpy


class sdataset:

    def __init__(self):
        self.a_dataset = None
        self.b_dataset = None
        self.c_dataset = None
        self.d_dataset = None

        self.a_labels = None
        self.b_labels = None
        self.c_labels = None
        self.d_labels = None



    def load(self, x, y):

        total_size = x.shape[0]
        chunk_size = total_size / 4
        indices = numpy.random.permutation(x.shape[0])

        chunk1_idx = indices[:chunk_size-1]
        chunk2_idx = indices[chunk_size:(2*chunk_size-1)]
        chunk3_idx = indices[2*chunk_size:(3 * chunk_size - 1)]
        chunk4_idx = indices[3 * chunk_size:]

        print(chunk1_idx)
        print(chunk2_idx)
        print(chunk3_idx)
        print(chunk4_idx)

        self.a_dataset = x[chunk1_idx, :]
        self.b_dataset = x[chunk2_idx, :]
        self.c_dataset = x[chunk3_idx, :]
        self.d_dataset = x[chunk4_idx, :]

        self.a_labels = y[chunk1_idx, :]
        self.b_labels = y[chunk2_idx, :]
        self.c_labels = y[chunk3_idx, :]
        self.d_labels = y[chunk4_idx, :]
