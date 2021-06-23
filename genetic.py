from statistics import median
import random
import numpy as np
from sklearn.preprocessing import normalize
from numpy import errstate,isneginf,array
from dca import estimate_precision, balanced_rearrangement_matrices, estimate_mean_precision, balanced_rearrangement_matrices_binary
import multiprocessing as mp
import ray

@ray.remote
def transform_estimate_mean_precision( matrix, data ):

    data_transpose = data.transpose()
    reduced_data = matrix @ data_transpose
    reduced_data = reduced_data.transpose()
    reduced_data = reduced_data - np.amin(reduced_data)
    new_matrix = normalize(reduced_data, norm='l1', axis=1)
    precision = estimate_mean_precision(new_matrix)

    return precision


# Takes data and a target dimension k
def genetic_algorithm(data, k):
    n = len(data[0])
    iterations = 10

    matrix_population = balanced_rearrangement_matrices(n, k)
    prev = matrix_population
    ray.init()

    for i in range(iterations):
        print ("start finding precision  ")
        dirichilet_correlation_futures = [ transform_estimate_mean_precision.remote( matrix , data ) for matrix in matrix_population ]
        dirichilet_correlation = ray.get( dirichilet_correlation_futures )
        print ( "iteration *************** ", i)
        print ( min(dirichilet_correlation ))

        print("end finding precision  ")
        updated_population = []
        median_dc = median(dirichilet_correlation)

        dc_updated = np.array(list((map(lambda x: min(x / median_dc, 1), dirichilet_correlation))))

        if np.all(dc_updated == 1):
            return prev

        with errstate(divide='ignore'):
            fitness = - 1 * np.log(dc_updated)
        br_matrices = [(fitness[i], matrix_population[i]) for i in range(len(fitness))]

        br_matrices = sorted(br_matrices, key=lambda x: x[0], reverse=True)
        #print ( br_matrices )

        # Take the first half to be added to the new population
        # Add to new population if fitness > 0
        for br_matrix in (br_matrices[:len(br_matrices) // 2]):
            # print ( br_matrix[0] )
            if br_matrix[0] > 0 and not np.all(br_matrix[1] == 0):
                updated_population.append(br_matrix)

        new_population = [mat for (fit_score, mat) in updated_population]
        new_fitness = [fit_score for (fit_score, mat) in updated_population]
        new_fitness = new_fitness / np.sum(new_fitness)

        # Mutation operation where two random matrices are choosen based their fitness ( probability = fitness score )
        # and used to create a new matrix which is added to the population
        children = []
        while ( len(children) + len(new_fitness) ) < 1000 :
            parent1 = np.random.choice(len(new_population), p=new_fitness)
            parent2 = np.random.choice(len(new_population), p=new_fitness)
            if updated_population[parent1][0] > 0 and updated_population[parent2][0] > 0:
                # Selecting the alpha and beta as 0.5
                child = (updated_population[parent1][1] * 0.5) + (updated_population[parent2][1] * 0.5)
                child = normalize(child, norm='l1', axis=0)

                if not np.all(child == 0):
                    children.append(child)
        # print ("Updated Population ", (updated_population[-1]))

        prev = matrix_population
        matrix_population = new_population + children

        if not any(fitness):
            return prev

        if not matrix_population:
            return prev

    ray.shutdown()

    return matrix_population
