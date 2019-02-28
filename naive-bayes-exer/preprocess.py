import re
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def data_preprocess(str_arg):
    """Accepts an unprocessed string

    Removes punctuation marks and common stopwords.
    Replaces multiple spaces with single spaces.
    Converts the words into lowercase.

    Returns a preprocessed string
    """

    cleaned_str = re.sub('[^a-z\s]+', ' ', str(str_arg), flags=re.IGNORECASE)
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)
    cleaned_str = ' '.join(
        [word for word in cleaned_str.split() if word not in stop_words])
    cleaned_str = cleaned_str.lower()

    return cleaned_str
