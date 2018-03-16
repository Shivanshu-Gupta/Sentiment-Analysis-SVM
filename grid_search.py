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
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# NOTE: Make sure to remove non-printable ascii characters using: 
# tr -cd '\11\12\15\40-\176' < file-with-binary-chars > clean-file

def load_data(train_file, test_file):
    train_data = [json.loads(line) for line in open(train_file).readlines()]
    test_data = [json.loads(line) for line in open(test_file).readlines()]
    
    train = {}
    train['review'] = [html.unescape(sample['reviewText']) for sample in train_data]
    train['summary'] = [html.unescape(sample['summary']) for sample in train_data]
    train['rating'] = np.array([sample['overall'] for sample in train_data])
    
    test = {}
    test['review'] = [html.unescape(sample['reviewText']) for sample in test_data]
    test['summary'] = [html.unescape(sample['summary']) for sample in test_data]
    test['rating'] = np.array([sample['overall'] for sample in test_data])
    
    classes = np.array([-1, 0, 1])

    def target(rating):
        if rating <= 2:
            return classes[0]
        elif rating == 3:
            return classes[1]
        else:
            return classes[2]
    train['target'] = np.array([target(rating) for rating in train['rating']])
    test['target'] = np.array([target(rating) for rating in test['rating']])

    return train, test, classes

def load_preprocessed_data(datafile):
    samples = [line.split('\t') for line in open(datafile).readlines()]
#     for i, sample in enumerate(samples):
#         if len(sample) != 3:
#             print('sample ' + str(i) + ' doesn\'t split to 3 values')
    samples = [sample for sample in samples if len(sample) == 3]
    X = pd.DataFrame({
        'review': [sample[0] for sample in samples],
        'summary': [sample[1] for sample in samples]})
    Y = [int(sample[2].strip()) for sample in samples]

    return X, Y

# train, test, classes = load_data('audio_train.json', 'audio_dev.json')
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

def display_results(preds, dev_target):
    n_correct = (preds == dev_target).sum()
    print("accuracy={:4.2f} ({}/{})".format(n_correct/preds.shape[0] * 100, n_correct, preds.shape[0]))
    print(confusion_matrix(dev_target, preds, labels=[-1, 0, 1]))
    print("macro-F1={:4.4f}".format(f1_score(dev_target, preds, labels=[-1, 0, 1], average='macro')))

def evaluate_feat_SGD(train_feat, train_target, dev_feat, dev_target, loss='hinge', penalty='l2', max_iter=None, average=True, alpha=0.0001, tol=1e-3):
    clf = SGDClassifier(loss=loss, penalty=penalty, random_state=2324, max_iter=max_iter, class_weight="balanced", average=average, alpha=alpha, tol=tol)
    clf.fit(train_feat, train_target)
    preds = clf.predict(dev_feat)
    display_results(preds, dev_target)
    return clf

def outputResults(accuracy, fscore, outfile, average, alpha, tol):
    rev_params = '- review: (ngram_range=(1,2), binary=False, max_df=0.8, min_df={})'.format(min_df)
    summ_params = '- summ: (ngram_range=(1,2), binary=False, max_df=0.8, min_df={})'.format(min_df)
    comb_params = 'f={}'.format(f)
    feat_params = rev_params + '\n' + summ_params + '\n' + comb_params
    
    optim_params = "SGD(loss='hinge', max_iter=1000, average={}, alpha={}, tol={})".format(average, alpha, tol)
    with open(outfile, 'a') as fout:
        fout.write(','.join([feat_params, optim_params, "{:4.4f}".format(fscore), "{:4.2f}".format(accuracy)]) + '\n')

def evaluate(args):
    average, alpha, tol = args
    clf = SGDClassifier(loss='hinge', penalty="l2", random_state=2324, max_iter=1000, 
                        class_weight="balanced", average=average, alpha=alpha, tol=tol)
    clf.fit(train_feat, train_target)
    preds = clf.predict(dev_feat)
    n_correct = (preds == dev_target).sum()
    accuracy = n_correct/preds.shape[0] * 100
    fscore = f1_score(dev_target, preds, labels=[-1, 0, 1], average='macro')
    print(','.join([str(min_df), str(f), str(average), str(alpha), str(tol)])) 
    print("accuracy={:4.2f} ({}/{})".format(n_correct/preds.shape[0] * 100, n_correct, preds.shape[0]))
    print(confusion_matrix(dev_target, preds, labels=[-1, 0, 1]))
    print("macro-F1={:4.4f}".format(f1_score(dev_target, preds, labels=[-1, 0, 1], average='macro')))
#     outputResults(accuracy, fscore, outfile, average, alpha, tol)
    with open(outfile, 'a') as fout:
        fout.write(','.join([str(min_df), str(f), str(average), str(alpha), str(tol), 
                             "{:4.4f}".format(fscore), "{:4.2f}".format(accuracy)]) + '\n')    

# create features
rev_vectorizer = SpacyCountVectorizer(ngram_range=(1,2), binary=False, max_df=0.8, min_df=1e-5)
train_rev_feat = rev_vectorizer.fit_transform(trainX['review'])
test_rev_feat = rev_vectorizer.transform(testX['review'])

summ_vectorizer = SpacyCountVectorizer(ngram_range=(1,2), binary=False, max_df=0.8, min_df=1e-5)
train_summ_feat = summ_vectorizer.fit_transform(trainX['summary'])
test_summ_feat = summ_vectorizer.transform(testX['summary'])

from multiprocessing import Pool
all_min_df = [5e-6, 1e-5, 2e-5, 4e-5]
all_f = [1, 1.5, 2, 3, 4]
for min_df in all_min_df[:2]:
    for f in all_f:
        train_feat = hstack([train_rev_feat, f * train_summ_feat])
        dev_feat = hstack([test_rev_feat, f * test_summ_feat])
        train_target = trainY
        dev_target = testY
        outfile = 'bigram_{}_{}.out'.format(min_df, f)
        settings = [(average, alpha, tol) for alpha in [5e-4, 1e-4, 2e-5] for tol in [1e-3, 1e-4, 1e-5] for average in [False, True]]
        pool = Pool(10)
        pool.map(evaluate, settings)
        pool.close()
        pool.join()
