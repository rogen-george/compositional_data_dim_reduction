from scipy.stats import dirichlet
from scipy.special import digamma
from scipy.special import polygamma
import numpy as np
import math
from numpy import errstate,isneginf,array

def estimate_precision( data ):
    # Fix the mean
    mean = [ 1 / len(data[0]) ] * len(data[0])

    dimensions = len(mean)
    N = len( data )
    p_k = np.zeros(dimensions)

    # Number of iterations for Newton Update
    iterations = 100

    # Calculate the value of log p_k
    for dimension in range( dimensions ):
        with errstate(divide='ignore'):
            p_k[dimension] = np.sum ( np.log( data[:, dimension] ) ) / N
        p_k[isneginf(p_k)] = -1000

    denominator = 0
    for i, m_k in enumerate(mean):
        denominator += m_k * ( p_k[i] - np.log(m_k) )

    # Calculate the initial value of precision s
    numerator = (dimensions - 1) / 2
    initial_s = - 1* numerator / denominator

    s = initial_s
    for i in range ( iterations ):
        #print (s)
        second_term = 0
        third_term = 0
        d2_term = 0
        for i, m_k in enumerate(mean):
            second_term += m_k * digamma( s * m_k )
            third_term += m_k * p_k[i]
            d2_term += ( (m_k) ** 2 ) * polygamma(1, s * m_k)

        d_log_likelihood =   N * ( digamma(s) - second_term + third_term )
        d2_log_likelihood =  N * ( polygamma(1, s) - d2_term )

        # Update the value of s after each iteration
        s = 1 / ( (1 / s) + (1 / (s ** 2) ) *  ( ( 1 / d2_log_likelihood ) * d_log_likelihood ) )

    return (s)

def balanced_rearrangement_matrices_binary(n, k):

    matrices = []

    while ( len(matrices) < 100 ):

        x = np.zeros((k , n))
        i = 0

        n_n = list(range(k))
        for i in range(n):

            #select one element to be 1
            a = int( np.random.choice(n_n , 1) )

            x[a][i] = 1
            #n_n.remove(a)
        matrices.append(x)
    return (matrices)


def balanced_rearrangement_matrices(n, k):

    matrices = []

    while ( len(matrices) < 1000 ):

        x = np.zeros((k , n))
        i = 0
        for i in range(n):

            x[:,i] = np.random.random(k)
            x[:,i]  =  ( x[:,i]  / np.sum( x[:,i] ) )

        matrices.append(x)
    return (matrices)


def estimate_mean_precision(data):
    # print ("here", data )

    # Initial s estimate
    s = np.sum(data[0])
    # Initial mean estimate
    mean = [1 / len(data[0])] * len(data[0])

    dimensions = len(data[0])
    N = len(data)
    p_k = np.zeros(dimensions)
    di_gamma = np.zeros(dimensions)

    # Calculate the value of log p_k
    for dimension in range(dimensions):
        with errstate(divide='ignore'):
            p_k[dimension] = np.sum(np.log(data[:, dimension])) / N
        p_k[isneginf(p_k)] = -1000

    # Number of iterations for Newton Update
    iterations = 10

    for b in range(10):
        # Fix the precision and estimate the mean

        for i in range(iterations):

            sum_term = 0
            for j in range(dimensions):
                sum_term += mean[j] * (p_k[j] - digamma(s * mean[j]))

            di_gamma = p_k - sum_term
            x = np.where(di_gamma >= -2.22, np.exp(di_gamma) + 1 / 2, -1 / (di_gamma - digamma(1)))
            # print ("printing digamma ", di_gamma)

            # Find Digamma inverse
            # Five Newton Iterations are enough to reach 14 points of precision
            for _ in range(10):
                x = x - np.divide((digamma(x) - di_gamma), polygamma(1, x))

            mean = x / np.sum(x)
            # print ("mean ", mean )

        # Now use this mean to estimate the precision s
        denominator = 0
        for i, m_k in enumerate(mean):
            denominator += m_k * (p_k[i] - math.log(m_k))

            # Calculate the initial value of precision s
        numerator = (dimensions - 1) / 2
        initial_s = -1 * numerator / denominator

        s = initial_s
        # print ("intial s", s)

        for i in range(iterations):

            # print (s)
            second_term = 0
            third_term = 0
            d2_term = 0
            for i, m_k in enumerate(mean):
                second_term += m_k * digamma(s * m_k)
                third_term += m_k * p_k[i]
                d2_term += ((m_k) ** 2) * polygamma(1, s * m_k)

            d_log_likelihood = N * (digamma(s) - second_term + third_term)
            d2_log_likelihood = N * (polygamma(1, s) - d2_term)

            # Update the value of s after each iteration
            s = 1 / ((1 / s) + ((1 / (s ** 2)) * ((1 / d2_log_likelihood) * d_log_likelihood)))

            # print ("s here ", s)
    return (s)
