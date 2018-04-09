import os
import argparse
import json
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


if __name__ == '__main__':
    train_data = args.train_data
    dev_data = args.dev_data
    save_dir = args.save_dir
    # load and vectorize train data
    print('vectorizing train data...')
    print('loading preprocessed train data...')
    train = load_preprocessed_data(train_data)
    print('fitting review vectorizer...')
    rev_vectorizer = SACountVectorizer(ngram_range=(1, 2), binary=False, max_df=0.8, min_df=1e-6)
    train_rev_feat = rev_vectorizer.fit_transform(train['review'])
    print('fitting summary vectorizer...')
    summ_vectorizer = SACountVectorizer(ngram_range=(1, 2), binary=False, max_df=0.8, min_df=1e-6)
    train_summ_feat = summ_vectorizer.fit_transform(train['summary'])
    trainX = hstack([train_rev_feat, 2 * train_summ_feat])
    trainY = train['target']
    print(trainX.shape)

    if dev_data:
        # load and vectorize dev data
        print('vectorizing dev data...')
        dev = load_preprocessed_data(dev_data)
        dev_rev_feat = rev_vectorizer.transform(dev['review'])
        dev_summ_feat = summ_vectorizer.transform(dev['summary'])
        devX = hstack([dev_rev_feat, 2 * dev_summ_feat])
        devY = dev['target']
        print(devX.shape)

    # train classifier
    print('training classifier...')
    clf = SGDClassifier(loss='hinge', penalty='l2', random_state=2324, class_weight="balanced", average=True, alpha=1e-4, tol=1e-3)
    clf.fit(trainX, trainY)
    if dev_data:
        # evaluate on dev set
        print('evaluating on dev data...')
        preds = clf.predict(devX)
        n_correct = (preds == devY).sum()
        print("accuracy={:4.2f} ({}/{})".format(n_correct / preds.shape[0] * 100, n_correct, preds.shape[0]))
        print(confusion_matrix(devY, preds, labels=[0, 1, 2]))
        print("macro-F1={:4.4f}".format(f1_score(devY, preds, labels=[0, 1, 2], average='macro')))
    # dump vectorizer and model
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print('saving review vectorizer...')
        joblib.dump(rev_vectorizer, save_dir + '/rev.pkl')
        print('saving summary vectorizer...')
        joblib.dump(summ_vectorizer, save_dir + '/summ.pkl')
        print('saving model...')
        joblib.dump(clf, save_dir + '/model.pkl')
