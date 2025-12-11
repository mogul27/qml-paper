from sklearn.svm import SVC
import numpy as np
from utilities import split_train_test_kernels, split_train_test_reduced

def qsvc_classify(evaluated_kernels: dict, y_train: np.array, y_test: np.array,
                  n_dimensions: tuple=(2, 4), classify_original: bool=False, 
                  original_dims: int=None, sr_datasets: dict=None):
    """
    Classify dataset using sklearn support vector classification. The kernels for
    the svc are the pre-computed quantum kernels.

    Parameters:
        evaluated_kernels (dict) -- Named kernels computed using quantum simulators or hardware
        y_train (Numpy array) -- Labels for training data used to fit svc
        y_test (Numpy array) -- Labels for test data used to score svc
        n_dimensions (tuple) -- Start/Stop value for number of reduced dimensions (Inclusive) (default: (2, 4))
        classify_original (bool) -- Whether to classify original dataset (default: False)
        original_dims (int) -- Dimensionality of original dataset (default: None)
        sr_datasets (dict): dict containing scaled dimensionality reduced datasets for use in quantum kernel (default: None)

    Returns:
        classification_accs (dict) -- Classification accuracy for quantum kernels on different dataset 
                                      dimensionality
    """

    train_kernels, test_kernels = split_train_test_kernels(evaluated_kernels)

    # Create counter to use enumerate for iterating over dimensionality and loop
    # progress seperately.
    dims = list(range(n_dimensions[0], n_dimensions[1]+1))

    # kernel dict keys are in order of dimension so use indexing to retrieve correct
    # kernel for dimensionality
    train_kernel_keys = list(train_kernels.keys())
    test_kernel_keys = list(test_kernels.keys())

    # Instantiate dictionary to hold scores from QSVC runs
    qsvc_scores = {}

    for count, dimensionality in enumerate(dims):

        svc = SVC(kernel='precomputed')
        svc.fit(train_kernels[train_kernel_keys[count]], y_train)
        svc_train_score = svc.score(train_kernels[train_kernel_keys[count]], y_train) 
        svc_score = svc.score(test_kernels[test_kernel_keys[count]], y_test)

        # Save score with feature map and dimensionality
        qsvc_name = train_kernel_keys[count].replace('_train', '')
        qsvc_scores[qsvc_name] = [svc_train_score, svc_score]
    
    if classify_original:

        svc = SVC(kernel='precomputed')
        # Original dataset appended so always last
        svc.fit(train_kernels[train_kernel_keys[-1]], y_train)
        svc_score = svc.score(test_kernels[test_kernel_keys[-1]], y_test)

        # Save score with feature map and dimensionality
        qsvc_name = train_kernel_keys[-1].replace('_train', '')
        qsvc_scores[qsvc_name] = svc_score

    return qsvc_scores

def run_classical_kernel(classical_kernel: str, y_train: np.array, y_test: np.array,
                         n_dimensions: tuple=(2, 4), sr_datasets: dict=None):
    """
    Run Support Vector Machine (SVM) using a classical kernel avaialble in sklearn
    to perform classification

    Parameters:
        classical kernel (str) -- Name of classical kernel function to use in SVM
        y_train (Numpy array) -- Labels for training data used to fit svc
        y_test (Numpy array) -- Labels for test data used to score svc
        n_dimensions (tuple) -- Start/Stop value for number of reduced dimensions (Inclusive) (default: (2, 4))
        sr_datasets (dict): dict containing scaled dimensionality reduced datasets for use in classical kernel (default: None)

    Returns:
        svc_scores (dict) -- Classification accuracy for classical kernel on different dataset dimensionality
    """

    train_data, test_data = split_train_test_reduced(sr_datasets)

    # Create counter to use enumerate for iterating over dimensionality and loop
    # progress seperately.
    dims = list(range(n_dimensions[0], n_dimensions[1]+1))

    # Perform the same operation as above for scaled_reduced datasets
    train_data_keys = list(train_data.keys())
    test_data_keys = list(test_data.keys())

    svc_scores = {}

    for count, dimensionality in enumerate(dims):
        classical_svc = SVC(kernel=classical_kernel)
        classical_svc.fit(train_data[train_data_keys[count]], y_train)
        classical_train_score = classical_svc.score(train_data[train_data_keys[count]], y_train) 
        classical_score = classical_svc.score(test_data[test_data_keys[count]], y_test)

        # save score for classical kernel with corresponding dimensionality
        svc_name = train_data_keys[count].replace('_train', '_'+classical_kernel)
        svc_scores[svc_name] = [classical_train_score, classical_score]

    return svc_scores

