import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from pprint import pprint
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, f1_score

parser = argparse.ArgumentParser(description='SVM-based Sentiment Analyzer')
parser.add_argument('--train_data', default='', required=True, metavar='PATH')
parser.add_argument('--dev_data', default='', metavar='PATH')
parser.add_argument('--save_dir', default='', metavar='PATH')
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


if __name__ == '__main__':
    # load and vectorize train data
    print('vectorizing train data...')
    print('loading preprocessed train data...')
    train, trainY = load_preprocessed_data(args.train_data)
    print('fitting review vectorizer...')
    rev_vectorizer = SpacyCountVectorizer(ngram_range=(1, 2), binary=False, max_df=0.8, min_df=1e-6)
    train_rev_feat = rev_vectorizer.fit_transform(train['review'])
    print('fitting summary vectorizer...')
    summ_vectorizer = SpacyCountVectorizer(ngram_range=(1, 2), binary=False, max_df=0.8, min_df=1e-6)
    train_summ_feat = summ_vectorizer.fit_transform(train['summary'])
    trainX = hstack([train_rev_feat, 2 * train_summ_feat])
    print(trainX.shape)

    if args.dev_data:
        # load and vectorize dev data
        print('vectorizing dev data...')
        dev, devY = load_preprocessed_data(args.dev_data)
        dev_rev_feat = rev_vectorizer.transform(dev['review'])
        dev_summ_feat = summ_vectorizer.transform(dev['summary'])
        devX = hstack([dev_rev_feat, 2 * dev_summ_feat])
        print(devX.shape)

    # train classifier
    print('training classifier...')
    clf = SGDClassifier(loss='hinge', penalty='l2', random_state=2324, class_weight="balanced", average=True, alpha=1e-4, tol=1e-3)
    clf.fit(trainX, trainY)
    if args.dev_data:
        # evaluate on dev set
        print('evaluating on dev data...')
        preds = clf.predict(devX)
        n_correct = (preds == devY).sum()
        print("accuracy={:4.2f} ({}/{})".format(n_correct / preds.shape[0] * 100, n_correct, preds.shape[0]))
        print(confusion_matrix(devY, preds, labels=[-1, 0, 1]))
        print("macro-F1={:4.4f}".format(f1_score(devY, preds, labels=[-1, 0, 1], average='macro')))
    # dump vectorizer and model
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print('saving review vectorizer...')
        joblib.dump(rev_vectorizer, args.save_dir + '/rev.pkl')
        print('saving summary vectorizer...')
        joblib.dump(summ_vectorizer, args.save_dir + '/summ.pkl')
        print('saving model...')
        joblib.dump(clf, args.save_dir + '/model.pkl')

