import pandas as pd
import numpy as np
import os
import sys
import getopt
from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayes

datafile = 'vocab.csv'


def main(argv):
    alpha = 0
    word_freq = 0
    try:
        opts, _ = getopt.getopt(argv, 'ha:w:', ["alpha=", "word_freq="])
    except getopt.GetoptError:
        print('Invalid argument')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('driver.py -a <alpha> -wf <word frequency>')
        elif opt in ('-a', '--alpha'):
            alpha = arg
        elif opt in ('-w', '--word_freq'):
            word_freq = arg

    return alpha, word_freq


if __name__ == '__main__':
    alpha, word_freq = main(sys.argv[1:])
    dataset = pd.read_csv(os.path.join(
        sys.path[0], datafile), sep=',', index_col=0, header=0)
    dataset.dropna(how='any', subset=['text'], inplace=True)
    print('----- Data Loaded -----')

    x_train, x_test, y_train, y_test = train_test_split(
        dataset['text'], dataset['ham_or_spam'],
        test_size=0.2, random_state=191, stratify=dataset['ham_or_spam'])

    classes = np.unique(y_train)

    nb = NaiveBayes(classes, float(alpha), int(word_freq))
    print('----- Training In Progress with alpha = {} -----'.format(alpha))
    nb.train(x_train, y_train)
    print('----- Training Completed with {} words -----'.format(
        nb.vocab_length))

    prob_classes = nb.test(x_test)
    test_acc = np.sum(prob_classes == y_test)/float(y_test.shape[0])

    print('Test Set Accuracy: {:.05%}'.format(test_acc))
