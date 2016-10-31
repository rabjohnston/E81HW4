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

    def first(self):
        return self._results[0]
    
    def get_test_pred(self, results, param):
        tp = {}
        for result in results:
            key = '{}'.format(result.params[param])
            tp[key] = result.test_preds

        return tp




