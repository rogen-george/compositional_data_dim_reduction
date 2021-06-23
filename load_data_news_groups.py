from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import pandas as pd

def get_data_news_groups():

    newsgroups_train = fetch_20newsgroups(subset='train', categories= ['misc.forsale', 'alt.atheism'])
    newsgroups_test = fetch_20newsgroups(subset='test', categories= ['misc.forsale', 'alt.atheism'])


    train_label = newsgroups_train.target
    test_label = newsgroups_test.target

    # Return the term document matrix
    vectorizer = CountVectorizer(stop_words = 'english', min_df=10, max_features=1425)

    # Divide by the total word count to change the data into compositional
    train_data = vectorizer.fit_transform(newsgroups_train.data).toarray()
    test_data = vectorizer.transform(newsgroups_test.data).toarray()

    n = len( train_data )
    m = len( test_data )
    train_data = train_data / train_data.sum( axis = 1).reshape(n,1)
    test_data = test_data / test_data.sum(axis = 1).reshape(m,1)

    validation_data = test_data[(m//2):]
    validation_label = test_label[(m//2):]

    test_data = test_data[:(m//2)]
    test_label = test_label[:(m//2)]

    positive_train = []
    negative_train = []
    # Split train data into positive and negative to solve the problem of class imbalances
    for i in range( len( train_data ) ):
        if train_label[i] == 1:
            positive_train.append( train_data[i] )
        else:
            negative_train.append( train_data[i] )

    positive_train = np.array( positive_train )
    negative_train = np.array( negative_train )

    return positive_train, negative_train, validation_data, validation_label, test_data, test_label