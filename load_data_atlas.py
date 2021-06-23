import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  svm_original import train_svm

from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn import svm
from sklearn.svm import LinearSVC




def shuffle_create_data( data_positive, data_negative ):

    data = np.append( data_positive, data_negative, axis = 0)
    label = np.append( np.ones( len( data_positive) ), np.zeros( len( data_negative) ), axis = 0)

    data_label = np.append( data, label.reshape(( len(label) ), 1 ), axis = 1 )
    np.random.shuffle( data_label )

    features = len( data_label[0] )
    data = data_label[:, 0:features - 1]
    label = data_label[:, features - 1]

    return data, label


def get_data_atlas():

    data_male = pd.read_excel( "data_central_europe.xlsx" )
    data_female = pd.read_excel( "data_scandinavia.xlsx" )


    positive_train = np.array( data_male[:60] )
    negative_train = np.array( data_female[:60] )

    validation_positive = np.array( data_male[60:80] )
    validation_negative = np.array( data_female[60:80] )

    test_positive = np.array( data_male[80:100] )
    test_negative = np.array( data_female[80:100] )

    validation_data, validation_label = shuffle_create_data( validation_positive, validation_negative)
    test_data, test_label = shuffle_create_data( test_positive, test_negative )

    return positive_train, negative_train, validation_data, validation_label, test_data, test_label



