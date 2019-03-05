# Description

There are two (2) Bayesian classifiers presented here:

- one in classifying emails as either ham or spam
- one in classifying tweets on nuclear energy as positive, negative, unrelated to nuclear energy, and information sharing only

## Exer 2A - Bayesian Classifier for Classifying E-mails as Ham or Spam

- Preprocessed dataset is at `ham_spam_dataset.csv`

## Exer 2B - Bayesian Classifier for Classifying Tweets on Nuclear Energy

- Preprocessed dataset is at `nuclear_senti_dataset.csv`

# Naive Bayes Classifier

This contains a manual implementation of a Naive Bayes Classifier.

The datasets, Ham and Spam and Nuclear Sentiment, used for the exercise can be obtained [here](https://drive.google.com/file/d/1gowKxNEdMOUWGWlKHUTd1HdT0hTnvP7w/view?usp=sharing) and [here](https://drive.google.com/file/d/12O49qYBm5Tp-f12BHoZ_cZwqNjGfIQxU/view?usp=sharing), respectively.

# Installation

`pip install -r requirements.txt`

# Usage

1. Download the datasets and place them within the `naive-bayes-exer` folder.
2. Open a terminal within the directory of the folder.
3. Run the following command.
   - `python <driver> -a <alpha> -r <reduce(T|F)> -v <vocab_count>`
     - `<driver>` refers to `ham_spam_driver.py` or `nuclear_senti_driver.py`
     - `<alpha>` is the value used for the Laplace smoothing
     - `<reduce(T|F)>` determines whether the vocabulary will be reduced or not
     - `<vocab_count>` is the amount of words to be left per class after reduction. **This must be set regardless of whether `<reduce(T|F)>` is true or false**

# Miscellaneous

Other files in the `naive-bayes-exer` folder

- `driver_helper.py` - provides handling of `argv` when driver files are ran in the terminal
- `naive_bayes.py` - implementation of the Naive Bayes learning algorithm, refer to the [Naive Bayes Classifier](#naive-bayes-classifier) section
- `preprocess.py` - used for preprocessing the text data in the datasets

Written reports for each exercise can be found in the `Written Reports` folder.

Link for this project can be found in [this](https://github.com/VJMendoza/cs-191-exercises) repository.
