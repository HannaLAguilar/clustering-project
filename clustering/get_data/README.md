# Preprocessing data

Preprocessing pipeline to go from raw data to processed data. 
Focus on transforming numerical data and categorical data,
without removing outliers. 

To this end, these algorithms were implemented:

## Numerical data:

- Standardization: StandardScaler. (*Standardize features by 
removing the mean and scaling to unit variance*.(Sklearn, 2022))

- Nan values: Replace with the median of the feature.

## Categorical data:

- Standardization: OneHorEncoder. (*Encodes categorical features as a 
single numeric array.* (Sklearn, 2022))

- Nan values: Replace with the highest frequency
value of the feature.
