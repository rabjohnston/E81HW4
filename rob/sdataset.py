import numpy

class sdataset():

    def load(self, x, y):

        chunk_size = x.shape[0] / 4
        indices = numpy.random.permutation(x.shape[0])
        chunk1_idx, chunk2_idx = indices[:80], indices[80:]
        chunk1_x, chunk2_x = x[chunk1_idx, :], x[chunk2_idx, :]
        chunk1_y, chunk2_y = y[chunk1_idx, :], x[chunk2_idx, :]