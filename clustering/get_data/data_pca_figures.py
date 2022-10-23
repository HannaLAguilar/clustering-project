import pandas as pd
from sklearn.decomposition import PCA

from clustering.visualization import visualize
from clustering.path_definitions import PROCESSED_DATA_PATH, ROOT_PATH


def process_pca(dataset_name: str):
    path = PROCESSED_DATA_PATH / dataset_name
    df = pd.read_csv(path, index_col=0)
    X = df.iloc[:, :-1]
    y = df['y_true']
    y.name = 'class'
    pca = PCA(2).fit(X)
    reduced_data = pca.transform(X)
    variance = pca.explained_variance_ratio_.cumsum()[1]
    return reduced_data, y, variance


if __name__ == '__main__':

    PATH_REPORT_FIGURES = ROOT_PATH / 'reports/figures'

    DATASET_NAME = ['iris.csv',
                    'vowel.csv',
                    'cmc.csv',
                    'pen-based.csv']

    for dataset_name in DATASET_NAME:
        raw_name = dataset_name.split('.')[0]
        REDUCED_DATA, Y, VAR = process_pca(dataset_name)
        fig = visualize.plot_pca(REDUCED_DATA, Y)

        fig.savefig(PATH_REPORT_FIGURES /
                    f'{raw_name}_pca.png')
