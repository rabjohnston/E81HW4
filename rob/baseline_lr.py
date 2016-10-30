from sklearn.linear_model import LogisticRegression
from dataset import Dataset

class Baseline_lr:

    def __init__(self, ds):
        self.name = 'LR'
        self.ds = ds

        self.test_accuracy = None

    def create(self):
        lr = LogisticRegression(C=0.01, n_jobs=-1).fit(self.ds.train_dataset, self.ds.train_labels)
        self.test_accuracy = lr.score(self.ds.test_dataset, self.ds.test_labels)
        self.lr.predict_proba()

