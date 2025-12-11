from sklearn.decomposition import PCA
from openTSNE import TSNE
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime

def obtain_reduced_datasets(method: str='pca', X_train: np.array=None, y_train: np.array=None,
                            X_test: np.array=None, n_dimensions: tuple=(2, 4),
                            save_dim_info: bool=False, tsne_arguments: dict=None, 
                            visualise_tsne_umap: bool=False, umap_arguments: dict=None):
    """
    Select and call chosen dimensionality reduction technique to obtain reduced dimension dataset
    Datasets returned with dimensionality reduced to all values between values supplied in n_dimensions (inclusive)

    Parameters:
        method (str): Dim reduction method to apply. (default: 'pca') (options: 'pca', 'tsne', 'umap')
        X_train (numpy array): Training features to fit to and transform (default: None)
        X_test (numpy array): Testing features to transform (default: None)
        y_train (numpy array): Labels for training dataset (default: None)
        n_dimensions (tuple): Start and stop values for dimensions to transform to (inclusive) (default: (2,4))
        save_dim_info (bool) -- If True and using lda or pca, print explained variance ratio and save graph (default: False)
        tnse_arguments (dict) -- Core adjustable parameters for tsne. Ignored if passed with non-matching dim-method (default: None)
        visualise_tsne_umap (bool) -- If True, save a plot of 2D projection. Requires 2D embedding is part of dimensionality 
                                      reduction. (default: False) 
        umap_arguments (dict) -- Core adjustable parameters for umap. Ignored if passed with non-matching dim-method (default: None)

    
    Returns:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
    """

    # Select dim reduction technique using case switch
    match method:
        case 'pca':
            reduced_datasets = instantiate_fit_pca(X_train=X_train, X_test=X_test,
                                                   n_dimensions=n_dimensions)

        case 'tsne':
            reduced_datasets = instantiate_fit_tsne(X_train=X_train, X_test=X_test,
                                                    n_dimensions=n_dimensions,
                                                    tsne_arguments=tsne_arguments)

        case 'umap':
            reduced_datasets = instantiate_fit_umap(X_train=X_train, X_test=X_test,
                                            n_dimensions=n_dimensions,
                                            umap_arguments=umap_arguments)
        
        case _:
            raise ValueError("dim_method must be one of 'pca', 'tsne', 'umap'")
        
    # Visualise PCa explained variance ratio
    if save_dim_info and method == 'pca':
        exp_variance_ratio = get_pca_info(train_set=X_train)
        print("Explained variance from each subsequent component:{}".format(exp_variance_ratio))
    
    # Visualise 2D embedding of data
    elif visualise_tsne_umap and method=='tsne':
        visualise_2d_tsne(reduced_datasets=reduced_datasets, y_train=y_train)

    elif visualise_tsne_umap and method=='umap':
        visualise_2d_umap(reduced_datasets=reduced_datasets, y_train=y_train)

    return reduced_datasets

def instantiate_fit_pca(X_train: np.array=None, X_test: np.array=None, n_dimensions: tuple=(2, 4)):
    """
    Fit Principal Component Analysis to train dataset and transform both train and test datasets.
    Datasets returned with dimensionality reduced to all values between values supplied in n_dimensions (inclusive)

    Parameters:
        X_train (numpy array): Training features to fit to and transform (default: None)
        X_test (numpy array): Testing features to transform (default: None)
        n_dimensions (tuple): Start and stop values for dimensions to transform to (inclusive) (default: (2,4))
    
    Returns:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
    """

    reduced_datasets = {}

    # PCA fit only to train data to avoid leaking information from test data
    for dimensionality in range(n_dimensions[0], n_dimensions[1]+1):
        pca = PCA(n_components=dimensionality)
        pca.fit(X_train)
        reduced_train = pca.transform(X_train)
        reduced_test = pca.transform(X_test)

        # Save reduced dimensionality datasets as 'n'd in dictionary
        reduced_datasets['X_train_{}d'.format(dimensionality)] = reduced_train
        reduced_datasets['X_test_{}d'.format(dimensionality)] = reduced_test

    return reduced_datasets        

def get_pca_info(train_set: np.array=None):
    """
    Perform PCA on full dataset to obtain information about how much variance is explained by each
    principal component and visualise. Information is saved into a png file to be viewed with the current
    time as id

    Parameters:
        train_set (numpy array) -- Training dataset to perform PCA on
    
    Returns:
        exp_variance_ratio (numpy array) -- Amount of variance explained by including each subsequent 
        principal component

    """

    pca = PCA()
    pca.fit_transform(train_set)
    # Variance explained by including each subsequent principal component
    exp_variance_ratio = pca.explained_variance_ratio_
    # Add in a value for 0 components for visualisation
    exp_variance_ratio = np.insert(exp_variance_ratio, 0, 0)

    plt.plot(np.cumsum(exp_variance_ratio))
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.xticks(np.arange(0, len(exp_variance_ratio), 1.0))
    plt.savefig("pca_plots/{}.png".format(str(datetime.datetime.now())))

    return exp_variance_ratio

def instantiate_fit_tsne(X_train: np.array=None, X_test: np.array=None, n_dimensions: tuple=(2, 4),
                         tsne_arguments: dict=None):
    """
    Fit t-SNE to train dataset and transform both train and test datasets.
    Datasets returned with dimensionality reduced to all values between values supplied in n_dimensions (inclusive)

    Parameters:
        X_train (numpy array): Training features to fit to and transform (default: None)
        X_test (numpy array): Testing features to transform (default: None)
        n_dimensions (tuple): Start and stop values for dimensions to transform to (inclusive) (default: (2,4))
        tsne_arguments (dict) -- Core adjutable arguments for opentsne t-SNE (default: None)
    
    Returns:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
    """

    # If optional arguments passed, set those for t-SNE instance
    if tsne_arguments:
        perplexity=tsne_arguments['perplexity']
        n_iter=tsne_arguments['n_iter']
        learning_rate=tsne_arguments['learning_rate']

    # Otherwise explicitly set to default values to those in opentsne TSNE documentation
    else:
        perplexity=30
        n_iter=500
        learning_rate='auto'

    reduced_datasets = {}

    # PCA fit only to train data to avoid leaking information from test data
    for dimensionality in range(n_dimensions[0], n_dimensions[1]+1):
        tsne = TSNE(n_components=dimensionality, perplexity=perplexity, n_iter=n_iter,
                    learning_rate=learning_rate, random_state=27)
        train_embedding = tsne.fit(X_train)
        reduced_train = train_embedding.transform(X_train)
        reduced_test = train_embedding.transform(X_test)

        # Save reduced dimensionality datasets as 'n'd in dictionary
        reduced_datasets['X_train_{}d'.format(dimensionality)] = reduced_train
        reduced_datasets['X_test_{}d'.format(dimensionality)] = reduced_test

    return reduced_datasets        

def visualise_2d_tsne(reduced_datasets: dict=None, y_train: np.array=None):
    """
    Visualise the 2d embedding of tsne to get an indication of whether tsne is 
    finding similar clusters well in 2-Dimensions. Dataset must have been reduced 
    to 2 dimensions as part of dimensionality reduction or error is raised.

    Parameters:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
        y_train (numpy array): Labels for training dataset

    Returns:
        None
    """

    try:
        train_2d = reduced_datasets['X_train_2d']
    except:
        raise ValueError("Cannot visualise t-SNE if dataset was not reduced to 2 dimensions")

    plot_df = pd.DataFrame(train_2d, columns=["Dim1", "Dim2"])
    plot_df["label"] = y_train

    scatterplot = sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue="label")
    plt.savefig("tsne_plots/{}.png".format(str(datetime.datetime.now())))

def instantiate_fit_umap(X_train: np.array=None, X_test: np.array=None, n_dimensions: tuple=(2, 4),
                         umap_arguments: dict=None):
    """
    Fit UAMP to train dataset and transform both train and test datasets.
    Datasets returned with dimensionality reduced to all values between values supplied in n_dimensions (inclusive)

    Parameters:
        X_train (numpy array): Training features to fit to and transform (default: None)
        X_test (numpy array): Testing features to transform (default: None)
        n_dimensions (tuple): Start and stop values for dimensions to transform to (inclusive) (default: (2,4))
        umap_arguments (dict) -- Core adjutable arguments for umap (default: None)
    
    Returns:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
    """

    # If optional arguments passed, set those for UMAP instance
    if umap_arguments:
        n_neighbors=umap_arguments['n_neighbors']
        min_dist=umap_arguments['min_dist']
        metric=umap_arguments['metric']

    # Otherwise explicitly set to default values to those in UMAP documentation
    else:
        n_neighbors=15
        min_dist=0.1
        metric='euclidean'

    reduced_datasets = {}

    # PCA fit only to train data to avoid leaking information from test data
    for dimensionality in range(n_dimensions[0], n_dimensions[1]+1):
        umap_reducer = umap.UMAP(n_components=dimensionality, n_neighbors=n_neighbors,
                    min_dist=min_dist, metric=metric, random_state=27)
        umap_reducer.fit(X_train)
        reduced_train = umap_reducer.transform(X_train)
        reduced_test = umap_reducer.transform(X_test)

        # Save reduced dimensionality datasets as 'n'd in dictionary
        reduced_datasets['X_train_{}d'.format(dimensionality)] = reduced_train
        reduced_datasets['X_test_{}d'.format(dimensionality)] = reduced_test

    return reduced_datasets        

def visualise_2d_umap(reduced_datasets: dict=None, y_train: np.array=None):
    """
    Visualise the 2d embedding of umap to get an indication of whether umap is 
    finding similar clusters well in 2-Dimensions. Dataset must have been reduced 
    to 2 dimensions as part of dimensionality reduction or error is raised.

    Parameters:
        reduced_datasets (dict): dict containing dimensionality reduced datasets
        y_train (numpy array): Labels for training dataset

    Returns:
        None
    """

    try:
        train_2d = reduced_datasets['X_train_2d']
    except:
        raise ValueError("Cannot visualise umap if dataset was not reduced to 2 dimensions")

    plot_df = pd.DataFrame(train_2d, columns=["Dim1", "Dim2"])
    plot_df["label"] = y_train

    scatterplot = sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue="label")
    plt.savefig("umap_plots/{}.png".format(str(datetime.datetime.now())))
