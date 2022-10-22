from sklearn import metrics

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


# N_cluster range for every dataset
n_clusters_iris = list(range(2, 7, 1))
n_clusters_vowel = list(range(2, 16, 1))
n_clusters_cmc = list(range(2, 10, 1))
n_clusters_pen = list(range(7, 16, 1))

# Real number of class
n_iris_real = 3
n_vowel_real = 11
n_cmc_real = 3
n_pen_real = 10
