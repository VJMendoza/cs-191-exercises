# Naive Bayes Classifier

This contains a manual implementation of a Naive Bayes Classifier.

The datasets, Ham and Spam and Nuclear Sentiment, used for the exercise can be obtained [here](https://drive.google.com/file/d/1gowKxNEdMOUWGWlKHUTd1HdT0hTnvP7w/view?usp=sharing) and [here](https://drive.google.com/file/d/12O49qYBm5Tp-f12BHoZ_cZwqNjGfIQxU/view?usp=sharing)

To run the exercise, run the following steps:
1. Download the datasets and place them within the naive-bayes-exer folder.
2. Open a terminal within the directory of the folder.
3. Run the following command.
    - `python <driver> -a <alpha> -r <reduce(T|F)> -v <vocab_count>`
        - `<driver>` refers to `ham_spam_driver.py` or `nuclear_senti_driver.py`
        - `<alpha>` is the value used for the Laplace smoothing
        - `<reduce(T|F)>` determines whether the vocabulary will be reduced or not
        - `<vocab_count>` is the amount of words to be left per class after reduction. **This must be set regardless of whether `<reduce(T|F)>` is true or false**