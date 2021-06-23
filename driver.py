from load_data import test_data, test_label, validation_data, validation_label, positive_train, negative_train
from load_data import sample_range, k_values, save_folder

from svm_original import train_svm
from genetic import genetic_algorithm
import random
import numpy as np
import matplotlib.pyplot as plt
from skbio.stats.composition import ilr, ilr_inv
from skbio.stats.composition import clr, clr_inv
from sklearn.decomposition import PCA
from coda import coda_pca_run
from sklearn.utils import shuffle
import pandas as pd
import pickle

def split_train_test( positive_train, negative_train, sample_size ):

    train_positive = positive_train[: ((sample_size // 2)) , :]
    train_postive_label = np.ones( ( sample_size // 2 )  )

    train_negative = negative_train[: (sample_size // 2) , :]
    train_negative_label = np.zeros( ( sample_size // 2 )  )

    train_sample_data = np.append( train_positive, train_negative, axis = 0)
    train_sample_label = np.append( train_postive_label, train_negative_label, axis = 0)

    data_label = np.append( train_sample_data, train_sample_label.reshape((( len(train_sample_label) ) , 1 )), axis = 1)
    np.random.shuffle( data_label )

    features = len(data_label[0])
    train_sample_data = data_label[:, 0:features - 1]
    train_sample_label = data_label[:,features - 1]

    return train_sample_data, train_sample_label

def dca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):

    br_matrices = genetic_algorithm(train_sample_data, reduce)

    dca_array = []
    for br_matrix in br_matrices:

        reduced_data = np.matmul(br_matrix, train_sample_data.transpose()).transpose()
        reduced_test = np.matmul(br_matrix, test_data.transpose()).transpose()
        reduced_validation = np.matmul(br_matrix, validation_data.transpose()).transpose()

        accuracy_dca, roc_dca_data = train_svm(reduced_data, train_sample_label, reduced_test, test_label,
                                              reduced_validation, validation_label)

        dca_array.append( (accuracy_dca, roc_dca_data) )
        dca_array = sorted( dca_array , key = lambda x : x[0], reverse= True)

    return dca_array[0][0], dca_array[0][1]


def pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):
    #clr_data_train = clr(train_sample_data)
    #clr_test = clr(test_data)

    #clr_data_validation = clr(validation_data)

    # Do PCA to reduce dimensions
    pca = PCA(n_components=reduce)
    fit_train = np.ascontiguousarray(pca.fit_transform(train_sample_data))
    fit_test = np.ascontiguousarray(pca.transform(test_data))
    fit_validation = np.ascontiguousarray(pca.transform(validation_data))

    pca_reduced_train = np.nan_to_num(fit_train)
    accuracy_pca, roc_pca_data = train_svm(pca_reduced_train, train_sample_label, fit_test, test_label,
                                                  fit_validation, validation_label)

    return accuracy_pca, roc_pca_data

def clr_pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):
    clr_data_train = clr(train_sample_data)
    clr_test = clr(test_data)

    clr_data_validation = clr(validation_data)

    # Do PCA to reduce dimensions
    pca = PCA(n_components=reduce)
    fit_train_clr = np.ascontiguousarray(pca.fit_transform(clr_data_train))
    fit_test_clr = np.ascontiguousarray(pca.transform(clr_test))
    fit_validation_clr = np.ascontiguousarray(pca.transform(clr_data_validation))

    pca_clr_reduced_train = np.nan_to_num(fit_train_clr)
    accuracy_clr, roc_pca_clr_data = train_svm(pca_clr_reduced_train, train_sample_label, fit_test_clr, test_label,
                                                  fit_validation_clr, validation_label)

    return accuracy_clr, roc_pca_clr_data

def ilr_pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):
    ilr_data_train = ilr( train_sample_data )
    ilr_test = ilr( test_data )

    ilr_data_validation = ilr(validation_data)
    # Do PCA to reduce dimensions
    pca = PCA(n_components=reduce)
    fit_train_ilr = np.ascontiguousarray( pca.fit_transform(ilr_data_train) )
    fit_test_ilr = np.ascontiguousarray( pca.transform(ilr_test) )
    fit_validation_ilr = np.ascontiguousarray( pca.transform(ilr_data_validation) )

    pca_ilr_reduced_train = np.nan_to_num( fit_train_ilr )
    accuracy_ilr, roc_pca_ilr_data = train_svm( pca_ilr_reduced_train, train_sample_label, fit_test_ilr, test_label, fit_validation_ilr, validation_label )

    return accuracy_ilr, roc_pca_ilr_data

def coda_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):
    coda_train, coda_validation, coda_test = coda_pca_run(reduce, train_sample_data, validation_data, test_data, "CODA_PCA",
                                                          learning_rate)

    coda_train = np.nan_to_num(coda_train)
    coda_validation = np.nan_to_num(coda_validation)
    coda_test = np.nan_to_num(coda_test)

    accuracy_coda, roc_coda = train_svm(coda_train, train_sample_label, coda_test, test_label, coda_validation,
                                  validation_label)
    return accuracy_coda, roc_coda

def coda_ae_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data ):
    coda_ae_train, coda_ae_validation, coda_ae_test = coda_pca_run(reduce, train_sample_data, validation_data, test_data,
                                                                   "CODA_AE", learning_rate)

    coda_ae_train = np.nan_to_num(coda_ae_train)
    coda_ae_validation = np.nan_to_num(coda_ae_validation)
    coda_ae_test = np.nan_to_num(coda_ae_test)

    accuracy_coda_ae, roc_coda_ae = train_svm(coda_ae_train, train_sample_label, coda_ae_test, test_label, coda_ae_validation,
                                        validation_label)
    return accuracy_coda_ae, roc_coda_ae

def get_array_mean( values, iterations ):
    values = np.array( values )
    values_mean = np.sum( values , axis = 0 ) / ( iterations + 1 )
    return values_mean


# Params :
# iterations : Number of times to run the algorithm to take average
# sample size : sample size of training data
# dca_reduce : The target dimension to reduce to
# train_data_svm : training data
# train_label_svm : training label


def train( train_sample_data, train_sample_label, reduce ):

    results_accuracy, results_roc = {}, {}
    results_accuracy['original_data'], results_roc['original_data'] = train_svm( train_sample_data, train_sample_label, test_data, test_label, validation_data, validation_label)

    # DCA transform the data and classify
    results_accuracy['dca'], results_roc['dca'] = dca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data )

    # Do ILR and CLR transformation
    # Set zeros to small values
    train_sample_data[train_sample_data == 0] = 0.1e-32
    test_data[test_data == 0] = 0.1e-32
    validation_data[validation_data == 0] = 0.1e-32

    #results_accuracy['pca'], results_roc['pca'] = pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data )

    #results_accuracy['clr'], results_roc['clr'] = clr_pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data )
    #results_accuracy['ilr'], results_roc['ilr'] = ilr_pca_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data )

    #results_accuracy['coda'], results_roc['coda'] = coda_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data)
    #results_accuracy['coda_ae'], results_roc['coda_ae'] = coda_ae_transform( reduce, train_sample_data, train_sample_label, validation_data, test_data )

    return results_accuracy, results_roc


learning_rate = 1e-3
iterations = 30
methods = ['original_data', 'dca']

results = {'accuracy': {} , 'roc': {} }
results_average = {'accuracy': {} , 'roc': {} }

for k in k_values:
    results['accuracy'][k] = {}
    results['roc'][k] = {}

    results_average['accuracy'][k] = {}
    results_average['roc'][k] = {}

    for method in methods:
        results['accuracy'][k][method] = []
        results['roc'][k][method] = []

        results_average['accuracy'][k][method] = []
        results_average['roc'][k][method] = []

for i in range(iterations):

    for method in methods:
        for k in k_values:
            results['accuracy'][k][method].append([])
            results['roc'][k][method].append([])

    for k in k_values:
        for sample_size in sample_range :
            train_sample_data, train_sample_label = split_train_test( positive_train, negative_train, sample_size )
            results_accuracy, results_roc = train( train_sample_data, train_sample_label , k )

            for method in methods:
                results['accuracy'][k][method][i].append( results_accuracy[method] )
                results['roc'][k][method][i].append( results_roc[method] )

    np.random.shuffle( positive_train )
    np.random.shuffle( negative_train )

    for k in k_values:
        for method in methods:
            results_average['accuracy'][k][method] = get_array_mean( results['accuracy'][k][method] , i )
            results_average['roc'][k][method] = get_array_mean( results['roc'][k][method] , i )

    # Save this data as a pickle file
    with open( save_folder +'/un_tuned_data_more_' + str(i) + '.pickle', 'wb') as handle:
        pickle.dump(results_average, handle, protocol=pickle.HIGHEST_PROTOCOL)
