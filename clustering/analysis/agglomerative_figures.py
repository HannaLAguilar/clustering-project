import pandas as pd

from clustering.analysis import definitions
from clustering.path_definitions import PROCESSED_DATA_PATH, ROOT_PATH
from clustering.visualization import visualize

PATH_REPORT_FIGURES = ROOT_PATH / 'reports/figures'


def main(dataset_name: str):
    raw_name = dataset_name.split('.')[0]
    n_classes = definitions.n_iris_real
    df = pd.read_csv(PROCESSED_DATA_PATH / f'agglo_results_{dataset_name}',
                     index_col=0)

    cases = {'affinity': ['euclidean', 'cosine'],
             'linkage': ['single', 'complete', 'average']}

    for affinity in cases['affinity']:
        for linkage in cases['linkage']:
            case = [affinity, linkage]
            title = f'{case[0].capitalize()} - {case[1]}'
            case_df = df[(df['affinity'] == case[0])
                         & (df['linkage'] == case[1])]

            # Internal index
            fig1 = visualize.plot_internal_index(case_df, n_classes, title)
            fig1.savefig(PATH_REPORT_FIGURES /
                         f'{raw_name}_{case[0]}_{case[1]}_internal.png')

            # External index
            fig2 = visualize.plot_external_index(case_df,
                                                 title)
            fig2.savefig(PATH_REPORT_FIGURES /
                         f'{raw_name}_{case[0]}_{case[1]}_external.png')


if __name__ == '__main__':
    DATASET_NAME = ['iris.csv',
                    'vowel.csv',
                    'cmc.csv',
                    'pen-based.csv']

    [main(name) for name in DATASET_NAME]
