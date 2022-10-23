import numpy as np
import pandas as pd
import itertools

import sys
sys.path.append('C:/Users/lgtuc/OneDrive/Documents/School/Masters/1st Semester/UB - Intro to Machine Learning/Clustering-Project')

from clustering.path_definitions import PROCESSED_DATA_PATH
from time import time
from typing import Dict, List, Tuple

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Internal metrics
INTERNAL_METRICS = {'calinski': metrics.calinski_harabasz_score,
                    'davies': metrics.davies_bouldin_score,
                    'silhouette': metrics.silhouette_score}

# External metrics
EXTERNAL_METRICS = {'ARI': metrics.adjusted_rand_score,
                    'AMI': metrics.adjusted_mutual_info_score,
                    'homo': metrics.homogeneity_score,
                    'compl': metrics.completeness_score,
                    'v-measure': metrics.v_measure_score}

def initial_u(c, n):
    u_matrix = np.zeros((c, n))
    cumulative_col_sums = np.zeros(n)
    for i in range(c):
        for j in range(n):
            # if only one cluster
            if c == 1:
                u_matrix[i][j] = 1
            # if first cluster in matrix
            elif i == 0:
                rand_val = np.random.random_sample()
                u_matrix[i][j] = rand_val
                cumulative_col_sums[j] = rand_val
            # if last cluster in matrix
            elif i == (c - 1):
                curr_col_sum = cumulative_col_sums[j]
                u_matrix[i][j] = 1 - curr_col_sum
            # all other clusters in matrix
            else:
                curr_col_sum = cumulative_col_sums[j]
                rand_val = np.random.uniform(0, 1 - curr_col_sum)
                u_matrix[i][j] = rand_val
                cumulative_col_sums[j] += rand_val

    return u_matrix


def update_centers(mem_matrix, data, m, c, p):
    denom_vals = np.zeros(len(mem_matrix)) # calculate a denominator for each cluster

    # calculate the denominator of Cluster Equation
    for i in range(len(mem_matrix)):
        for j in range(len(mem_matrix[0])):
            squared_val = mem_matrix[i][j] ** m
            denom_vals[i] += squared_val

    #  sum each element column-wise per cluster -- this will give the numerator for that cluster/feature pair
    #  and then divide that by the denom_values number for the cluster -- this will give the cluster center value
    #  I think this should be an array of size c x p
    numerators = []
    for cluster in range(c):
        numer_sums = np.zeros(p)
        for x in range(len(mem_matrix[0])):
            numer_terms = np.multiply(mem_matrix[cluster][x] ** m, data[x][:])
            numer_sums = np.add(numer_sums, numer_terms)
        numerators.append(numer_sums)

    v_matrix = [numerators[i] / denom_vals[i] for i in range(len(mem_matrix))]

    return v_matrix


def find_distances(v_matrix, data):
    '''distance matrix of dimensions n x c -- is distance
    between each x's data point and the cluster center'''
    distance_matrix = [[np.linalg.norm(point - cluster) for cluster in v_matrix] for point in data]
    return distance_matrix


def update_u_matrix(distance_matrix, mem_matrix):
    '''calculate new weights using the distances in the distance matrix '''
    new_u = [[] for i in range(len(mem_matrix))]
    for row in np.array(distance_matrix):
        for cluster in range(len(mem_matrix)):
            new_u[cluster].append(row[cluster] ** -1 / np.sum(np.power(row, -1)))
    return new_u


def FCM(data, c, m, num_iters, term_threshold, limit):
    n = data.shape[0]
    p = data.shape[1]
    iter = 0
    no_change = 0
    past_centers = np.ones((c,p))

    mem_matrix = initial_u(c, n)

    while num_iters != -1:
        iter += 1
        v_matrix = update_centers(mem_matrix, data, m, c, p)
        distance_matrix = find_distances(v_matrix, data)
        mem_matrix = update_u_matrix(distance_matrix, mem_matrix)

        if (abs(np.sum(np.array(past_centers) - np.array(v_matrix)))) < term_threshold:
            no_change += 1
        else:
            no_change = 0
        past_centers = v_matrix.copy()
        if iter == num_iters:
            num_iters = -1
            labels = np.argmax(mem_matrix, 0)
            print('Max iterations reached')
        elif no_change == limit:
            num_iters = -1
            labels = np.argmax(mem_matrix, 0)
            print('Number of iterations at convergence: {}'.format(iter))

    return v_matrix, labels

def hyperparameter_clustering(X: np.ndarray,
                              y_true_num: np.ndarray,
                              parameters: Dict,
                              internal_metrics: Dict =
                              INTERNAL_METRICS.values(),
                              external_metrics: Dict =
                              EXTERNAL_METRICS.values()) -> Tuple[List, List, float]:

    t0 = time()
    da = []
    params_combs = list(itertools.product(*list(parameters.values())))
    param = {}
    KMD_clustering = []
    for param_comb in params_combs:
        for index, p in enumerate(list(parameters.keys())):
            param[p] = param_comb[index]

        t0 = time()
        # Perform clustering
        centers, labels = FCM(X.values, param['c'], param['m'], param['num_iters'], param['term_threshold'], param['limit'])
        tf = time() - t0

        KMD_clustering.append(labels)
        # Save in a list
        result = [tf, *param_comb]
        # Internal index metrics
        result += [m(X, labels)
                   for m in internal_metrics]
        # External index metrics
        result += [m(y_true_num, labels)
                   for m in external_metrics]
        da.append(result)

    func_time = time() - t0
    return da, KMD_clustering, func_time


def main(data_name: str, save: bool = True):
    # Data
    path = PROCESSED_DATA_PATH / data_name
    df = pd.read_csv(path, index_col=0)

    X = df.iloc[:, :-1]

    y_true = df['y_true']
    le = LabelEncoder().fit(y_true.values)
    y_true_num = le.transform(y_true)

    # Parameters for cluster
    params = {'c': [2, 3, 4, 5, 6],
              'm': [2],
              'num_iters': [100],
              'term_threshold': [0.000001],
              'limit':[10]}

    # Perform sensitive analysis
    da = hyperparameter_clustering(X, y_true_num, params)
    metric_data, clus, global_time = da
    columns = ['time', *list(params.keys())]
    columns = columns + list(INTERNAL_METRICS.keys()) + list(
        EXTERNAL_METRICS.keys())

    # # Metric dataset
    metric_df = pd.DataFrame(metric_data, columns=columns)

    if save:
        metric_df.to_csv(PROCESSED_DATA_PATH / f'fcm_results_{data_name}')

    return metric_df, global_time


if __name__ == '__main__':
    DATASET_NAME = 'cmc.csv'
    METRICS_DF, GLOBAL_TIME = main(DATASET_NAME)
    print(METRICS_DF)
    print(GLOBAL_TIME)