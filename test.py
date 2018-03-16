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
    samples = [line.split('\t') for line in open(datafile).readlines()]
    samples = [sample for sample in samples if len(sample) == 3]
    X = pd.DataFrame({
        'review': [sample[0] for sample in samples],
        'summary': [sample[1] for sample in samples]})
    Y = [int(sample[2].strip()) for sample in samples]

    return X, Y


# Define vectorizers
class SpacyCountVectorizer(CountVectorizer):

    def __init__(self, lowercase=True, ngram_range=(1, 1), binary=False, vocabulary=None,
                 max_features=None, max_df=1.0, min_df=1, pos=True):
        super(SpacyCountVectorizer, self).__init__(lowercase=lowercase, ngram_range=ngram_range, binary=binary, vocabulary=vocabulary,
                                                   max_features=max_features, max_df=max_df, min_df=min_df)
        self.pos = pos

    def tokenize(self, doc):
        if doc == '':
            return []
        features = doc.split('  ')
        if self.pos:
            return features
        else:
            return [feat.split(':|:')[0] for feat in features]

    def build_tokenizer(self):
        return lambda doc: self.tokenize(doc)


class ReviewExtractor(object):

    def transform(self, X):
        return X['review']

    def fit(self, X, y=None):
        return self


class SummaryExtractor(object):

    def transform(self, X):
        return X['summ']

    def fit(self, X, y=None):
        return self


def writePreds(output_file, preds):
    classmap = {-1: 1, 0: 3, 1: 5}
    with open(output_file, 'w') as outf:
        for pred in preds:
            outf.write(str(classmap[pred]) + '\n')


if __name__ == '__main__':
    print('loading review vectorizer...')
    rev_vectorizer = joblib.load(args.load_dir + '/rev.pkl')
    print('loading summary vectorizer...')
    summ_vectorizer = joblib.load(args.load_dir + '/summ.pkl')
    print('loading model...')
    clf = joblib.load(args.load_dir + '/model.pkl')

    # load and vectorize test data
    print('loading preprocessed test data...')
    test, testY = load_preprocessed_data(args.test_data)
    print('vectorizing test data...')
    test_rev_feat = rev_vectorizer.transform(test['review'])
    test_summ_feat = summ_vectorizer.transform(test['summary'])
    testX = hstack([test_rev_feat, 2 * test_summ_feat])
    print(testX.shape)

    # evaluate on test set
    print('running evaluation...')
    preds = clf.predict(testX)
    n_correct = (preds == testY).sum()
    print("accuracy={:4.2f} ({}/{})".format(n_correct / preds.shape[0] * 100, n_correct, preds.shape[0]))
    print(confusion_matrix(testY, preds, labels=[-1, 0, 1]))
    print("macro-F1={:4.4f}".format(f1_score(testY, preds, labels=[-1, 0, 1], average='macro')))
    if args.output_file:
        writePreds(args.output_file, preds)

