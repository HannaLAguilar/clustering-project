from typing import Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.io import arff

from clustering.path_definitions import RAW_DATA_PATH, PROCESSED_DATA_PATH


def import_data(p: Union[Path, str]) -> Tuple[pd.DataFrame,
                                              arff.arffread.MetaData]:
    data, meta = arff.loadarff(p)
    df = pd.DataFrame(data)
    return df, meta


def remove_predicted_value(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:, :-1]


def check_null_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def standardization(df: pd.DataFrame) -> np.ndarray:
    try:
        df['ca'] = df['ca'].astype('object')  # Only for heart-c dataset
    except:
        pass
    num_features = df.select_dtypes(include=np.number).columns
    num_transformer = Pipeline(steps=[
        ('replace_nan', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    cat_features = df.select_dtypes(exclude=np.number).columns
    cat_transformer = Pipeline(steps=[
        ('replace_nan', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())])

    ct = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)])

    X_trans = ct.fit_transform(df)
    return X_trans


def preprocessing(data_name: str, save: bool = True) -> pd.DataFrame:
    # Preprocessing
    print(f'---Preprocessing {data_name} dataset---')
    data_path = RAW_DATA_PATH / data_name
    df, meta = import_data(data_path)
    df = remove_predicted_value(df)
    columns = df.columns
    nulls = check_null_values(df)
    if nulls.sum() != 0:
        print(f'There is nulls values: {nulls}')
    else:
        print(f'Nan values: 0')

    X = standardization(df)
    # Save
    if save:
        df = pd.DataFrame(X, columns=columns)
        df.to_csv(PROCESSED_DATA_PATH / data_name)
    return df


if __name__ == '__main__':
    DATASET_PATH = ['vowel.arff', 'pen-based.arff', 'mushroom.arff']
    PROCESSED_DATASETS = [preprocessing(name) for name in DATASET_PATH]
