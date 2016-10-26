import os
from result import Result

class Results():

    def __init__(self):
        self._results = []


    def load(self, dir):

        for file in os.listdir(dir):
            if file.endswith('.params'):
                r = Result()
                basefile = file[:-7]
                r.load('{}/{}'.format(dir, basefile))

                self._results.append(r)


    def subset(self, d1, d2):
        return all((k in d2 and d2[k] == v) for k, v in d1.items())
        #return all(item in superset.items() for item in subset.items())


    def get_results(self, search_params):

        all_params = []

        # Search for all items that have params that equal search_params
        for result in self._results:
            if self.subset( search_params, result.params):
                all_params.append(result)

        return all_params

    def highest(self, search_param):
        best_result = self._results[0]
        high_val = best_result.params[search_param]

        for result in self._results:
            if result.params[search_param] > high_val:
                best_result = result
                high_val = best_result.params[search_param]

        return best_result

    def get_test_pred(self, results, param):
        tp = {}
        for result in results:
            key = '{}'.format(result.params[param])
            tp[key] = result.test_preds

        return tp



def main():
    """
    Unit test code for Results class
    :return:
    """
    dir = 'results/NN20161025'

    # Load in the results
    rs = Results()
    rs.load(dir)

    # Search for all results that had a batch_size of 256
    search1 = {'batch_size': 256}
    find = rs.get_results(search1)
    for f in find:
        print('Result 1: ', f.params)

    # Search for all results that had a batch_size of 256, hidden_nodes of 2048 and lamb_reg of 0.005
    search2 = {'batch_size': 256, 'hidden_nodes': 2048, 'lamb_reg': 0.005}
    find = rs.get_results(search2)
    for f in find:
        print('Result 2: ', f.params)

    # For printing the ROC curves we get a dictionary of predictions for where the hidden_nodes rate is the only
    # unconstrained hyper-parameter
    # Search for all results that had a batch_size of 256, hidden_nodes of 2048 and lamb_reg of 0.005
    search3 = {'batch_size': 256, 'lamb_reg': 0.005}
    find = rs.get_results(search3)
    for f in find:
        print('Result 3: ', f.params)

    tp = rs.get_test_pred(find, 'hidden_nodes')

    for item in tp:
        print('Key: ', item)


    # Find the best accuracy
    print('Highest accuracy: ', rs.highest('accuracy').params)


if __name__ == '__main__':
    main()
