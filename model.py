import json
import logging
import spacy
import html
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from time import time
from pprint import pprint
from tabulate import tabulate
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Load data
def load_preprocessed_data(datafile):
    samples = [line.split('\t') for line in open(datafile).readlines()]
    samples = [sample for sample in samples if len(sample) == 3]
    X = pd.DataFrame({
        'review': [sample[0] for sample in samples],
        'summary': [sample[1] for sample in samples]})
    Y = [int(sample[2].strip()) for sample in samples]

    return X, Y

trainX, trainY = load_preprocessed_data('train_tok_clean.txt')
testX, testY = load_preprocessed_data('test_tok_clean.txt')

# Define vectorizers
class SpacyCountVectorizer(CountVectorizer):
    def __init__(self, lowercase=True, ngram_range=(1,1), binary=False, vocabulary=None, 
                 max_features=None, max_df=1.0, min_df=1, pos=True):
        super(SpacyCountVectorizer, self).__init__(lowercase=lowercase, ngram_range=ngram_range, binary=binary, vocabulary=vocabulary,
                                                   max_features=max_features, max_df=max_df, min_df=min_df)
        self.pos = pos
    def tokenize(self, doc):
        if doc == '':
            return []
        if self.pos:
            return doc.split('  ')
        else:
            return [tok.split(':|:')[0] for feat in features]
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

# create features
rev_vectorizer = SpacyCountVectorizer(ngram_range=(1,2), binary=False, max_df=0.8, min_df=5e-6)
train_rev_feat = rev_vectorizer.fit_transform(trainX['review'])
test_rev_feat = rev_vectorizer.transform(testX['review'])

summ_vectorizer = SpacyCountVectorizer(ngram_range=(1,2), binary=False, max_df=0.8, min_df=5e-6)
train_summ_feat = summ_vectorizer.fit_transform(trainX['summary'])
test_summ_feat = summ_vectorizer.transform(testX['summary'])

train_feat1 = hstack([train_rev_feat, 3 * train_summ_feat])
test_feat1 = hstack([test_rev_feat, 3 * test_summ_feat])

# set features
train_feat = train_feat1; test_feat = test_feat1
print(train_feat.shape)
print(test_feat.shape)

clf = SGDClassifier(loss=loss, penalty=penalty, random_state=2324, class_weight="balanced", average=True, alpha=1e-4, tol=1e-5)
clf.fit(train_feat, train_target)
preds = clf.predict(test_feat)
n_correct = (preds == test_target).sum()
print("accuracy={:4.2f} ({}/{})".format(n_correct/preds.shape[0] * 100, n_correct, preds.shape[0]))
print(confusion_matrix(test_target, preds, labels=[-1, 0, 1]))
print("macro-F1={:4.4f}".format(f1_score(test_target, preds, labels=[-1, 0, 1], average='macro')))
