# Analysis

Analyzing different cluster algorithms by modifying their parameters and 
measuring performance with external and internal index metrics.

## Cluster Algorithms
- Agglomerative algorithm
- Mean shift
- Kmeans
- Bisecting Kmeans
- Kmedians
- Fuzzy C-Means

## Metrics

Perform different internal and external validation.

### Internal index
This metric allows to define the best cluster number (k) based on the mean intra-cluster distance and the mean nearest-cluster distance.

- Inertia (for Kmeans and derivatives)
- Calinski-Harabasz 
- Davies&-Bouldin
- Silhouette

### External index

This metrics compute the differences between *y_pred*, and a given *y_true*.

- Adjusted Rand-Index
- Adjusted Mutual
- Homogeneity
- Completeness
- V-measure
