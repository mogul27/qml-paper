import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_dataset(dataset: str=""):
    """
    Load dataset from data folder

    Parameters:
        dataset (str) -- Name of csv to load from data folder (default: "").

    Returns:
        data (pandas Dataframe) -- Dataframe of loaded dataset.
    """
    data = pd.read_csv("data/{}".format(dataset))

    return data
    
def split_dataset(dataset: pd.DataFrame, test_size: float=0.3, random_state: int=27,
                  sample_size: int=None):
    """
    Split a loaded dataframe into stratified X and Y for train and test sets.

    Parameters:
        dataset (pandas Dataframe) -- Loaded dataset to split.
        test_size -- Proportion of loaded dataset to use as held-out test set.
        random_state -- Controls shuffling before split. Used for repeatability.
        sample_size (int) -- Number of datapoints from full dataset to take as a sample. If not set, then full dataset is
        used. Raises ValueError if sample size > size of full dataset (default: None)
    
    Returns:
        X_train (numpy array) -- feature array for training
        X_test (numpy array) -- feature array for testing
        y_train (numpy array) -- target array for training
        y_test (numpy array) -- target array for testing
    """

    # Target variable is final column in all datasets
    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset[dataset.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y,
                                                        random_state=random_state)
    
    if sample_size:
        # Get size of train and tet
        train_datapoints = X_train.shape[0]
        test_datapoints= X_test.shape[0]

        # Sample size must be < total datapoints in full dataset
        if (sample_size < train_datapoints) and (sample_size < test_datapoints):
            X_train = X_train[:sample_size]
            X_test = X_test[:sample_size]
            y_train = y_train[:sample_size]
            y_test = y_test[:sample_size]

        else:
            raise ValueError("Sample size must have fewer datapoints than the dataset used")
        
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def apply_scaler(X_train: np.array, X_test: np.array, scaler: str='standard', minmaxvals: tuple=(0, 1)):
    """
    Fit an sklearn scaler to training data then transform both train and test data
    
    Parameters:
        X_train (numpy array) -- feature array to be scaled
        X_test (numpy array) -- feature array to be scaled
        scaler (str) -- Type of scaler to be applied (default: 'standard')
        minmaxvals (tuple) -- Maximum and Minimum values to scale data to if using MinMaxScaler (default: (0, 1))

    Returns:
        scaled_train (numpy array) -- scaled training data
        scaled_test (numpy array) -- scaled test data
    """

    # Create extendable lookup tble of scalers from sklearn
    scaler_lookup = {'standard': StandardScaler(), 'minmax': MinMaxScaler(feature_range=minmaxvals)}
    chosen_scaler = scaler_lookup[scaler]

    # Fit scaler on train data and transform test to avoid leaking
    fitted_scaler = chosen_scaler.fit(X_train)
    scaled_train = fitted_scaler.transform(X_train)
    scaled_test = fitted_scaler.transform(X_test)

    return scaled_train, scaled_test

def scale_reduced_datasets(reduced_datasets: dict=None, scaler:str='standard', min_max_vals: tuple=(0, 1)):
    """
    Apply scaling to the reduced datasets held in a dictionary

    Parameters:
        reduced_datasets (dict) -- Dictionary containing dimensionality reduced datasets (default: None)
        scaler (str) -- Type of scaler to be applied (default: 'standard')
        minmaxvals (tuple) -- Maximum and Minimum values to scale data to if using MinMaxScaler (default: (0, 1))

    Returns:
        sr_datasets (dict) -- Dictionary containing scaled reduced datasets
    """

    # Iterate over keys in the dictionary and select matching train and test
    # key pairs 
    dataset_keys = list(reduced_datasets.keys())
    train_test_pairs = [(dataset_keys[i], dataset_keys[i+1]) for i in 
                            range(0, len(dataset_keys), 2)]

    # Create dictionary to hold reduced datasets once they have been scaled
    sr_datasets = {}

    # Obtain reduced datasets with scaling applied
    for pair in train_test_pairs:
        # Retrieve matching train and test datasets and apply scaling
        X_train = reduced_datasets[pair[0]]
        X_test = reduced_datasets[pair[1]]
        scaled_X_train, scaled_X_test = apply_scaler(X_train=X_train, X_test=X_test, scaler=scaler, 
                                                     minmaxvals=min_max_vals)
        sr_datasets[pair[0]] = scaled_X_train
        sr_datasets[pair[1]] = scaled_X_test

    return sr_datasets

def include_original_dataset(reduced_datasets: dict, X_train: np.array, X_test: np.array):
    """
    Append original (non-reduced) dataset to the reduced datasets

    Parameters:
        reduced_datasets (dict) -- Dictionary containing dimensionality-reduced datasets
        X_train (Numpy array) -- Training feature data
        X_test (Numpy array) -- Testing feature data

    Returns
        reduced_datasets (dict) -- reduced_datasets with the original datasets added
        original_dims -- Dimensionality of original datasets
    """

    # Append original dataset to reduced datasets
    reduced_datasets['original_train'] = X_train
    reduced_datasets['original_test'] = X_test

    original_dims = X_train.shape[1]

    return reduced_datasets, original_dims




