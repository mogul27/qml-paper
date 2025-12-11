import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from data_processing import load_dataset
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import LabelEncoder

# Standardize features
scaler = StandardScaler()


def preprocess(dataset):
        # Target variable is final column in all datasets
    X = dataset.drop(dataset.columns[-1], axis=1)
    y = dataset[dataset.columns[-1]]

    X_scaled = scaler.fit_transform(X)
    return np.array(X_scaled), np.array(y)

# Fisher's Discriminant Ratio (FDR) calculation
def fishers_discriminant_ratio(dataset):
    X, y = preprocess(dataset)
    classes = np.unique(y)
    mean_vectors = []
    for cls in classes:
        mean_vectors.append(np.mean(X[y == cls], axis=0))

    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))  # Between-class scatter matrix
    S_W = np.zeros((X.shape[1], X.shape[1]))  # Within-class scatter matrix

    for cls, mean_vec in zip(classes, mean_vectors):
        n = X[y == cls].shape[0]
        mean_diff = (mean_vec - overall_mean).reshape(-1, 1)
        S_B += n * (mean_diff).dot(mean_diff.T)
        class_scatter = np.cov(X[y == cls], rowvar=False) * (n - 1)
        S_W += class_scatter

    # Calculate Fisher's discriminant ratio
    S_W_inv = np.linalg.pinv(S_W)
    fdr_matrix = S_W_inv.dot(S_B)
    fdr_value = np.trace(fdr_matrix)

    return fdr_value


# # k-NN Borderline Analysis
# def knn_borderline_analysis(dataset, k=5):

#     X, y = preprocess(dataset)
#     knn = NearestNeighbors(n_neighbors=k)
#     knn.fit(X)

#     borderline_count = 0
#     for i, point in enumerate(X):
#         neighbors = knn.kneighbors([point], return_distance=False)[0]
#         neighbor_labels = y[neighbors]
#         # Count the number of neighbors that belong to a different class
#         different_class_neighbors = (neighbor_labels != y[i]).sum()
        
#         # Consider the point borderline if more than half of its neighbors are different class
#         if different_class_neighbors > k / 2:
#             borderline_count += 1

#     fraction_borderline_points = borderline_count / len(X)
#     return fraction_borderline_points

def fraction_of_border_points_mst(dataset):
    """
    Calculate the fraction of well-defined border points using a Minimum Spanning Tree (MST).
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target vector.
    
    Returns:
    float: Fraction of points near the decision boundary.
    """
    X, y = preprocess(dataset)
    # Encode the class labels if they are not numeric
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Compute the distance matrix
    dist_matrix = distance_matrix(X, X)
    
    # Build the MST from the distance matrix
    mst = minimum_spanning_tree(dist_matrix).toarray().astype(float)
    
    # Find the edges that connect points of different classes
    border_edges = []
    for i in range(len(X)):
        for j in range(len(X)):
            if mst[i, j] > 0 and y[i] != y[j]:
                border_edges.append((i, j))
    
    # Identify unique border points
    border_points = set([point for edge in border_edges for point in edge])
    
    # Calculate the fraction of border points
    fraction_border_points = len(border_points) / len(X)
    
    return fraction_border_points

if __name__ == "__main__":
    # datasets = ['complex.csv']
    datasets = ['digits_2.csv', 'mushroom.csv', 'forest_cov_2.csv', 'car.csv', "complex.csv"]
    #datasets = ['car.csv']
    #datasets = ["forest_cov_1008.csv", "complex.csv", "car_766.csv", "digits_2_350.csv", "mushroom_934.csv"]

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        preprocess(dataset)
        borderline_fraction = fraction_of_border_points_mst(dataset)
        print(f"Fraction of borderline points for {dataset_name}: {borderline_fraction:.4f}")

        fdr = fishers_discriminant_ratio(dataset)
        print(f"Fisher's Discriminant Ratio for {dataset_name}: {fdr:.4f}")
        samples = dataset.shape[0]
        dim = dataset.shape[1] - 1 # -1 for target
        print(f"{dataset_name}: Samples: {samples}, Dims {dim}")
        
