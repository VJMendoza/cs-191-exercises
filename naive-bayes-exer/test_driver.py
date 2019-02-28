import numpy as np
from sklearn.datasets import fetch_20newsgroups

from naive_bayes import NaiveBayes

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

train_data = newsgroups_train.data  # getting all trainign examples
train_labels = newsgroups_train.target  # getting training labels

nb = NaiveBayes(np.unique(train_labels))
print("---------------- Training In Progress --------------------")

# start tarining by calling the train function
nb.train(train_data, train_labels)
print('----------------- Training Completed ---------------------')

newsgroups_test = fetch_20newsgroups(
    subset='test', categories=categories)  # loading test data
test_data = newsgroups_test.data  # get test set examples
test_labels = newsgroups_test.target  # get test set labels

pclasses = nb.test(test_data)

test_acc = np.sum(pclasses == test_labels)/float(test_labels.shape[0])

# Outputs : Test Set Examples:  1502
print("Test Set Examples: ", test_labels.shape[0])
# Outputs : Test Set Accuracy:  93.8748335553 %
print("Test Set Accuracy: ", test_acc*100, "%")
