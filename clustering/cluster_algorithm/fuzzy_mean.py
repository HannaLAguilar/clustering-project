import numpy as np
import pandas as pd
from clustering.path_definitions import PROCESSED_DATA_PATH


# initialize U matrix with random membership values
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


def update_centers(mem_matrix, data, m):
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
        numer_sums = np.zeros(c)
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
    new_u = [[], []]
    for row in distance_matrix:
        for cluster in range(len(mem_matrix)):
            new_u[cluster].append(row[cluster] ** -1 / np.sum(np.power(row, -1)))
    return new_u


def FCM(data, c, m, num_iters, term_threshold, limit):
    n = data.shape[0]
    p = data.shape[1]
    iter = 0
    no_change = 0
    past_centers = np.ones((c,n))

    mem_matrix = initial_u(c, n)
    while num_iters != -1:
        iter += 1
        v_matrix = update_centers(mem_matrix, data, m)
        distance_matrix = find_distances(v_matrix, data)
        mem_matrix = update_u_matrix(distance_matrix, mem_matrix)

        if (abs(np.sum(past_centers- v_matrix))) < term_threshold:
            no_change += 1
        else:
            no_change = 0
        past_centers = v_matrix.copy()
        if iter == num_iters:
            num_iters = -1
            labels = np.argmax(mem_matrix, 0)
        elif no_change == limit:
            num_iters = -1
            labels = np.argmax(mem_matrix, 0)

    return v_matrix, labels



path = PROCESSED_DATA_PATH / 'pen-based.csv'
pen_based_df = pd.read_csv(path,  index_col=0)


# define number of clusters
c = 2   # using 2 here for now for simplicity

# define fuzziness degree
m = 2

# determine number of feature vectors
n = pen_based_df.shape[0]

# determine number of features
p = pen_based_df.shape[1]

num_iterations = 100
term_threshold = 0.01

# initial U matrix and V matrix
# prev_u_matrix = initial_u()
# prev_v_matrix = initial_v()
# for t in range(num_iterations):
#     u_matrix = update_u_matrix()
#     v_matrix = update_clusters()


