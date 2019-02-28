import pandas as pd
import numpy as np
from collections import defaultdict

from preprocess import data_preprocess


class NaiveBayes:
    def __init__(self, classes, alpha):
        """Constructor takes in the number of classes of the training set

        classes is the number of classes in the training set ie Yes/No
        bow_dicts contains the words found in the training set
        such that the words are under their corresponding BoW
        """
        self.classes = classes
        self.bow_dicts = np.array([defaultdict(lambda:0)
                                   for index in range(self.classes.shape[0])])
        self.alpha = alpha

    def add_to_BoW(self, text, bow_number):
        """Accepts a preprocessed string that
        contains the words from training set

        Splits the words and adds the resulting token
        to its corresponding dict/BoW
        """

        if isinstance(text, np.ndarray):
            text = text[0]

        for token in text.split():
            self.bow_dicts[bow_number][token] += 1

    def train(self, dataset, labels):
        """Accepts a dataset with the shape (l x d)
        where l is the number of classes and the labels with a shape of (l)

        Training function for the Naive Bayes Model
        Computes for the BoW for each class
        """

        self.dataset = dataset
        self.labels = labels

        if not isinstance(self.dataset, np.ndarray):
            self.dataset = np.array(self.dataset)

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        for cat_index, category in enumerate(self.classes):
            # get all data for that category
            all_cat_data = self.dataset[self.labels == category]

            # clean the gathered data
            cleaned_data = [data_preprocess(cat_data)
                            for cat_data in all_cat_data]

            cleaned_data = pd.DataFrame(data=cleaned_data)

            # construct the BoW for that category
            np.apply_along_axis(self.add_to_BoW, 1, cleaned_data, cat_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])

        for cat_index, category in enumerate(self.classes):
            # Compute for probability of a category, p(C)
            prob_classes[cat_index] = np.sum(
                self.labels == category)/float(self.classes.shape[0])

            # Compute for total count of all words in each category
            # count = list(self.bow_dicts[cat_index].values())
            # removed +1
            cat_word_counts[cat_index] = np.sum(
                np.array(list(self.bow_dicts[cat_index].values())))
            + self.alpha

            # Get all words for this category
            all_words += self.bow_dicts[cat_index].keys()

        # Construct the vocab for the training set
        self.vocab = np.unique(np.array(all_words))
        self.vocab_length = self.vocab.shape[0]

        # Get all denominators per category
        # removed + 1 from self.vocab_length
        demons = np.array([cat_word_counts[cat_index] +
                           self.vocab_length + self.alpha for cat_index,
                           category in enumerate(self.classes)])

        # Compile the data into tuples
        self.cat_infos = [(self.bow_dicts[cat_index],
                           prob_classes[cat_index],
                           demons[cat_index]) for cat_index,
                          category in enumerate(self.classes)]

        self.cat_infos = np.array(self.cat_infos)

    def get_test_prob(self, data):
        """Accepts a single data entry

        Computes for the posterior probability of the given data entry
        """
        likelihood_prob = np.zeros(self.classes.shape[0])

        for cat_index, _ in enumerate(self.classes):
            for token in data.split():
                # Get the total count of this token from the training dict
                # removed +1
                token_counts = self.cat_infos[cat_index][0].get(token, 0)
                # Get the probability of this token
                token_prob = token_counts + self.alpha / \
                    float(self.cat_infos[cat_index][2]) + self.alpha

                # Store the probability
                likelihood_prob[cat_index] += np.log(token_prob)

        # Calculate for the posterior probability
        post_prob = np.empty(self.classes.shape[0])
        for cat_index, _ in enumerate(self.classes):
            post_prob[cat_index] = likelihood_prob[cat_index] + \
                np.log(self.cat_infos[cat_index][1])

        return post_prob

    def test(self, dataset):
        """Accepts a dataset for testing

        Calculates the probility of the test set against all categories and
        predicts the label of the test set
        """

        predictions = []
        for data in dataset:
            # Clean the test data entry
            cleaned_data = data_preprocess(data)
            # Compute for the posterior probability of the entry
            post_prob = self.get_test_prob(cleaned_data)
            # Store the label of the entry into predictions
            predictions.append(self.classes[np.argmax(post_prob)])

        return np.array(predictions)
