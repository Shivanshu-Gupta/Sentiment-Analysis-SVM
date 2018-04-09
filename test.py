import json
import argparse
import pandas as pd
from scipy.sparse import hstack
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score

parser = argparse.ArgumentParser(description='SVM-based Sentiment Analyzer')
parser.add_argument('--test_data', default='', required=True, metavar='PATH')
parser.add_argument('--load_dir', default='', required=True, metavar='PATH')
parser.add_argument('--output_file', default='', metavar='PATH')
args = parser.parse_args()


def load_preprocessed_data(datafile):
    data = [json.loads(line) for line in open(datafile)]
    X = pd.DataFrame({
        'review': [sample['review'] for sample in data],
        'summary': [sample['summary'] for sample in data],
        'target': [sample['target'] for sample in data]})
    return X


# Define vectorizers
class SACountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stop_words = self.get_stop_words()
        return lambda doc: self._word_ngrams(doc, stop_words)


def writePreds(output_file, preds):
    with open(output_file, 'w') as outf:
        for pred in preds:
            outf.write(str(2 * pred + 1) + '\n')


def display_results(preds, targets):
    n_correct = (preds == targets).sum()
    print("accuracy={:4.2f} ({}/{})".format(n_correct / preds.shape[0] * 100, n_correct, preds.shape[0]))
    print(confusion_matrix(targets, preds, labels=[0, 1, 2]))
    print("macro-F1={:4.4f}".format(f1_score(targets, preds, labels=[0, 1, 2], average='macro')))


if __name__ == '__main__':
    load_dir = args.load_dir
    print('loading review vectorizer...')
    rev_vectorizer = joblib.load(load_dir + '/rev.pkl')
    print('loading summary vectorizer...')
    summ_vectorizer = joblib.load(load_dir + '/summ.pkl')
    print('loading model...')
    clf = joblib.load(load_dir + '/model.pkl')

    # load and vectorize test data
    print('loading preprocessed test data...')
    test = load_preprocessed_data(args.test_data)
    print('vectorizing test data...')
    test_rev_feat = rev_vectorizer.transform(test['review'])
    test_summ_feat = summ_vectorizer.transform(test['summary'])
    testX = hstack([test_rev_feat, 2 * test_summ_feat])
    testY = test['target']
    print(testX.shape)

    # evaluate on test set
    print('running evaluation...')
    preds = clf.predict(testX)
    display_results(preds, testY)
    if args.output_file:
        writePreds(args.output_file, preds)
