import os
import argparse
import json
import spacy
import pickle
import html
import numpy as np

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

parser = argparse.ArgumentParser(description='SVM-based Sentiment Analyzer')
parser.add_argument('--input_file', default='', required=True, metavar='PATH')
parser.add_argument('--output_file', default='', required=True, metavar='PATH')
parser.add_argument('--pickle_file', default='', metavar='PATH')
args = parser.parse_args()


def load_data(datafile):
    samples = [json.loads(line) for line in open(datafile).readlines()]
    data = {}
    data['review'] = [html.unescape(sample['reviewText']) for sample in samples]
    data['summary'] = [html.unescape(sample['summary']) for sample in samples]
    data['rating'] = np.array([sample['overall'] for sample in samples])

    classes = np.array([-1, 0, 1])

    def target(rating):
        if rating <= 2:
            return classes[0]
        elif rating == 3:
            return classes[1]
        else:
            return classes[2]
    data['target'] = np.array([target(rating) for rating in data['rating']])

    return data


def preprocess(data, outfile, picklefile=None):
    # if picklefile:
    #     if not os.path.exists(picklefile):
    #         docs = list(nlp.pipe(data['review']))
    #         with open(picklefile, 'wb') as f:
    #             pickle.dump(docs, f)
    #     else:
    #         docs = pickle.load(open(picklefile, 'rb'))
    with open(outfile, 'w') as outf:
        review_docs = nlp.pipe(data['review'])
        summ_docs = nlp.pipe(data['summary'])
        for i, (review, summ, target) in enumerate(zip(review_docs, summ_docs, data['target'])):
            review = '  '.join([tok.text + ':|:' + tok.tag_ for tok in review if not tok.is_stop and tok.text.strip() != ''])
            summ = '  '.join([tok.text + ':|:' + tok.tag_ for tok in summ if not tok.is_stop and tok.text.strip() != ''])
            outf.write(review + '\t' + summ + '\t' + str(target) + '\n')
            if i % 1000 == 0:
                print(i)


if __name__ == '__main__':
    data = load_data(args.input_file)
    preprocess(data, args.output_file, args.pickle_file)

