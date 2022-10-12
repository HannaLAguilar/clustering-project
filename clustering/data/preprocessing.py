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


def string_decode(df: pd.DataFrame):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return df


def import_data(p: Union[Path, str]) -> Tuple[pd.DataFrame,
                                              arff.arffread.MetaData]:
    data, meta = arff.loadarff(p)
    df = pd.DataFrame(data)
    df = string_decode(df)
    return df, meta


def remove_predicted_value(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[:, :-1]


def check_null_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def standardization(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    # numerical features
    num_features = df.select_dtypes(include=np.number).columns
    num_transformer = Pipeline(steps=[
        ('replace_nan', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    # categorical features
    cat_features = df.select_dtypes(exclude=np.number).columns
    cat_transformer = Pipeline(steps=[
        ('replace_nan', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())])

    # transform columns
    ct = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)])
    X_trans = ct.fit_transform(df)

    # dataset cases
    # case 1: categorical and numerical features
    if len(cat_features) != 0 and len(num_features) != 0:
        columns_encoder = ct.transformers_[1][1]['encoder'].\
            get_feature_names_out(cat_features)
        columns = num_features.union(pd.Index(columns_encoder), sort=False)

    # case 2: only categorical features
    elif len(cat_features) != 0 and len(num_features) == 0:
        columns = ct.transformers_[1][1]['encoder'].\
            get_feature_names_out(cat_features)
        columns = pd.Index(columns)
        X_trans = X_trans.toarray()

    # case 3: only numerical features
    elif len(cat_features) == 0 and len(num_features) != 0:
        columns = num_features

    # catch an error
    else:
        print('There is a problem with features')

    # processed dataset
    processed_df = pd.DataFrame(X_trans, columns=columns)
    return processed_df


def preprocessing(data_name: str, save: bool = True) -> pd.DataFrame:
    # Preprocessing
    print(f'---Preprocessing {data_name} dataset---')
    data_path = RAW_DATA_PATH / data_name
    df, meta = import_data(data_path)
    df = remove_predicted_value(df)
    nulls = check_null_values(df)
    if nulls.sum() != 0:
        print(f'There is nulls values: {nulls}')
    else:
        print(f'Nan values: 0')

    process_df = standardization(df)
    # Save
    if save:
        process_df.to_csv(PROCESSED_DATA_PATH / data_name)
    return process_df


if __name__ == '__main__':
    DATASET_PATH = ['mushroom.arff', 'vowel.arff', 'pen-based.arff', ]
    PROCESSED_DATASETS = [preprocessing(name) for name in DATASET_PATH]
