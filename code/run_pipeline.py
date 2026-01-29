from utilities import qiskit_login
from data_processing import load_dataset, split_dataset, apply_scaler, scale_reduced_datasets, include_original_dataset
from dim_reduction import obtain_reduced_datasets
from feature_maps_kernels import create_pauli_kernels_sim, evaluate_pauli_kernel_hw, evaluate_pauli_kernel_sim
import numpy as np
from utilities import write_dict
from qsvc import qsvc_classify, run_classical_kernel
import datetime


def run_pipeline(filename: str=None, dim_method: str='pca', minmax_val_list: list=[(0, 1), None],
                 n_dimensions: tuple=(2,4), save_dim_info:bool=False,
                 tsne_arguments: dict=None, visualise_tsne_umap: bool=False,
                 umap_arguments: dict=None, post_pca_scale: bool=True, scalers: list=['standard', 'standard'],
                 feature_map_str: list=['Z', 'ZZ'], fm_arguments: dict=None, save_circuit: bool=False,
                 sampler_options: dict=None, fidelity_options: dict=None, simulator: bool=True,
                 qiskit_api_token:str=None, qiskit_channel='ibm_quantum', backend_override: str=None,
                 min_qubits: int=2, sample_size: int=None, classify_original: bool=False):
    """
    Main function to run kernel creation and QSVM classification pipeline

    Parameters:
        filename (str) -- Filename of csv to load from data folder (default: None)
        dim_method (str) -- Dim reduction method to apply (default: 'pca') (options: 'pca', 'tsne', 'umap')
        minmaxval_list (list) -- Min and max scaling values if MinMax scaler is selected  (default: [(0, 1), None])
        save_dim_info (bool) -- If True and using pca, print explained variance ratio and save graph (default: False)
        n_dimensions (tuple) -- Start/Stop value for number of reduced dimensions (Inclusive) (default: 2, 4)
        tsne_arguments (dict) -- Core adjustable parameters for tsne. Ignored if passed with non-matching dim-method (default: None)
        visualise_tsne_umap (bool) -- If True, save a plot of 2D projection. Requires that 2D embedding is part of dimensionality 
                                      reduction. (default: False)
        umap_arguments (dict) -- Core adjustable parameters for umap. Ignored if passed with non-matching dim-method (default: None)
        post_pca_scale (bool) -- Whether to scale the datasets after performing PCA prior to classification (default: True)
        scalers (list) -- Scalers to use on first and second scaling (default: ['standard', 'standard']) (options: 'standard', 'minmax')
        feature_map_str (str) -- Pauli string for Feature map to create. (default: ['Z', 'ZZ'])
        fm_arguments (dict) -- Optional arguments to creation of PauliFeatureMap (default: None)
        save_circuit (bool) -- Save quantum circuit layout drawing to fm_circuits folder (default: False)
        sampler_options (dict) -- Options to be passed to the creation of a Sampler instance (default: None)
        fidelity_options (dict) -- Options passed to ComputeUncompute fidelity instance (default: None)
        simulator (bool) -- If True, run circuits on local quantum simulator using primitives sampler or if False,
                            run circuits on an IBM quantum machine using Qiskit Runtime. (default: False)
        qiskit_api_token (str) -- API login token from IBM Quantum (default: None)
        qiskit_channel (str) -- Type of account to use to access IBM Quantum (default: 'ibm_quantum')
        backend_override (str) -- String representation of a backend to be used rather than the choseen
                                  optimal backend (default: None)
        min_qubits (int) -- The minimum number of qubits in a quantum backend when selecting optimal backend (default: 2)
        sample_size (int) -- Number of datapoints from full dataset to take as a sample. If not set, then full dataset is
                             used. Raises ValueError if sample size > size of full dataset (default: None)
        classify_original (bool) -- Whether to classify the original dataset along with the reduced versions (default: False)

    Returns:
        results (dict) -- Dictionary containing pipeline run names as keys and classification results as values
    """

    # Load and split dataset 
    dataset = load_dataset(dataset=filename)
    print("Dataset loaded: {}".format(filename))
    X_train, X_test, y_train, y_test = split_dataset(dataset, sample_size=sample_size)

    # Scale dataset before PCA
    scaled_X_train, scaled_X_test = apply_scaler(X_train=X_train, X_test=X_test, scaler=scalers[0],
                                                 minmaxvals=minmax_val_list[0])

    # Perform dimensionality reduction and optionally save pca information about component variance
    reduced_datasets = obtain_reduced_datasets(method=dim_method, X_train=scaled_X_train, X_test=scaled_X_test,
                                               y_train=y_train, n_dimensions=n_dimensions, save_dim_info=save_dim_info, 
                                               tsne_arguments=tsne_arguments, visualise_tsne_umap=visualise_tsne_umap,
                                               umap_arguments=umap_arguments)
    
    # Append the datasets and get dimensionality of datase  if also classifying non-reduced dataset
    if classify_original:
        reduced_datasets, original_dims = include_original_dataset(reduced_datasets, X_train, X_test)
    else:
        original_dims = None
        
    # Optionally apply scaling before quantum kernel
    if post_pca_scale:
        sr_datasets = scale_reduced_datasets(reduced_datasets, scaler=scalers[1],
                                             min_max_vals=minmax_val_list[1])
    else:
        sr_datasets = reduced_datasets
    
    print("Reduced datasets obtained: {}".format(list(sr_datasets.keys())))

    # Create kernels using Pauli feature maps for specified settings
    if simulator:
        # Classify using a quantum simulator
        kernels = create_pauli_kernels_sim(feature_map_str=feature_map_str, fm_arguments=fm_arguments,
                                       save_circuit=save_circuit, sampler_options=sampler_options,
                                       fidelity_options=fidelity_options, n_dimensions=n_dimensions,
                                       classify_original=classify_original, original_dims=original_dims)

        print("Kernel creation complete with the following kernels: {}".format(list(kernels.keys())))
        
        evaluated_kernels = evaluate_pauli_kernel_sim(kernel_dict=kernels, sr_datasets=sr_datasets, n_dimensions=n_dimensions
                                                      , classify_original=classify_original, original_dims=original_dims)
        
    else:
        # Login to Qiskit Runtime
        service = qiskit_login(token=qiskit_api_token, channel=qiskit_channel)
        # Create and evaluate pauli kernels with quantum hardware
        evaluated_kernels = evaluate_pauli_kernel_hw(feature_map_str=feature_map_str, fm_arguments=fm_arguments,
                                       save_circuit=save_circuit, sampler_options=sampler_options,
                                       fidelity_options=fidelity_options, n_dimensions=n_dimensions,
                                       service=service, sr_datasets=sr_datasets, min_qubits=min_qubits)
        
    print("Kernel evaluation complete with the following kernels: {}".format(list(evaluated_kernels.keys())))

    results = qsvc_classify(evaluated_kernels=evaluated_kernels, y_train=y_train, y_test=y_test,
                            n_dimensions=n_dimensions, classify_original=classify_original,
                            original_dims=original_dims, sr_datasets=sr_datasets)
    
    return results

def run_classical_pipeline(filename: str=None, dim_method: str='pca', minmax_val_list: list=[(0, 1), None],
                 n_dimensions: tuple=(2,4), tsne_arguments: dict=None, visualise_tsne_umap: bool=False,
                 umap_arguments: dict=None, post_pca_scale: bool=False, scalers: list=['standard', 'standard'],
                 classical_kernel: str=None):
    """
    Main function to run kernel creation and SVM classification pipeline for a classical (non-quantum) kernel 

    Parameters:
        filename (str) -- Filename of csv to load from data folder (default: None)
        dim_method (str) -- Dime reduction method to apply (default: 'pca') (options: 'pca', 'tsne', 'umap')
        minmaxval_list (list) -- Min and max scaling values for 2 rounds of scaling if MinMax scaler is selected  (default: [(0, 1), None])
        n_dimensions (tuple) -- Start/Stop value for number of reduced dimensions (Inclusive) (default: 2, 4)
        tsne_arguments (dict) -- Core adjustable parameters for tsne. Ignored if passed with non-matching dim-method (default: None)
        visualise_tsne_umap (bool) -- If True, save a plot of 2D projection. Requires 2D embedding is part of dimensionality 
                                      reduction. (default: False)
        umap_arguments (dict) -- Core adjustable parameters for umap. Ignored if passed with non-matching dim-method (default: None)
        post_pca_scale (bool) -- Whether to scale the datasets after performing PCA prior to classification (default: True)
        scalers (list) -- Scalers to use on first and second scaling (default: ['standard', 'standard']) (options: 'standard', 'minmax')
        classical_kernel (str) -- Select which of the sklearn built-in classical kernels to datasets (default: 'rbf')

    Returns:
        results (dict) -- Dictionary containing pipeline run names as keys and classification results as values
    """

    # Load and split dataset 
    dataset = load_dataset(dataset=filename)
    print("Dataset loaded: {}".format(filename))
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    # Scale dataset before PCA
    scaled_X_train, scaled_X_test = apply_scaler(X_train=X_train, X_test=X_test, scaler=scalers[0],
                                                 minmaxvals=minmax_val_list[0])

    # Perform dimensionality reduction and optionally save pca information about component variance ratioo
    reduced_datasets = obtain_reduced_datasets(method=dim_method, X_train=scaled_X_train, X_test=scaled_X_test,
                                               y_train=y_train, n_dimensions=n_dimensions, tsne_arguments=tsne_arguments, 
                                               visualise_tsne_umap=visualise_tsne_umap, umap_arguments=umap_arguments)

    # Optionally apply scaling before quantum kernel
    if post_pca_scale:
        sr_datasets = scale_reduced_datasets(reduced_datasets, scaler=scalers[1],
                                             min_max_vals=minmax_val_list[1])
    else:
        sr_datasets = reduced_datasets
    
    print("Reduced datasets for classical pipeline obtained: {}".format(list(sr_datasets.keys())))

    results = run_classical_kernel(classical_kernel=classical_kernel, y_train=y_train, y_test=y_test,
                                   n_dimensions=n_dimensions, sr_datasets=sr_datasets)
    
    return results

if __name__=="__main__":

    print("Pipeline starting...")

    # Set loop over entanglement method
    # entanglement_methods = ["linear", "full", "sca", "circular"]
    #neighbours = [3]
    neighbours = [3, 5, 10, 25, 35, 45]
    # Set loop over datasets
    datasets = ['digits_2.csv', 'mushroom.csv', 'forest_cov_2.csv']
    #datasets = ['digits_2.csv']
    #datasets = ['car.csv']
    #datasets = ["digits_2_350.csv", "mushroom_934.csv", "forest_cov_1008.csv", "complex.csv", "car_766.csv",]
    # datasets = ["digits_2_350.csv", "mushroom_934.csv"]
    
    # Car csv clasified seperately due to different number of max dimensions
    # datasets = ['car.csv']
    
    for neighbour in neighbours:
        for dataset in datasets:

        # Set variables and run pipeline
            filename = dataset
            classical_kernel='rbf'
            dim_method='umap'
            minmax_val_list = [(0, 1), (0, 1)]
            n_dimensions = (2, 7)
            tsne_arguments = {"n_iter": 500, "perplexity": 30,
                              "learning_rate": 'auto'}
            visualise_tsne_umap = False
            umap_arguments = {"n_neighbors": neighbour, "min_dist": 0.1,
                              "metric":'euclidean'}
            post_pca_scale = True
            scalers = ['standard', 'minmax']

            # Run function to obtain results for classical kernel
            results = run_classical_pipeline(filename=filename, classical_kernel=classical_kernel, dim_method=dim_method, 
                                            minmax_val_list=minmax_val_list, n_dimensions=n_dimensions, tsne_arguments=tsne_arguments,
                                            visualise_tsne_umap=visualise_tsne_umap, umap_arguments=umap_arguments,
                                            post_pca_scale=post_pca_scale, scalers=scalers)
            
            # Pass parameters to dictionary to record settings for pipeline run
            parameters_dict = {"filename": filename, "dim_method": dim_method, 
                            "classical_kernel": classical_kernel, "minmax_val_list": minmax_val_list,
                            "n_dimensions": n_dimensions, "tsne_arguments": tsne_arguments,
                            "visualise_tsne_umap": visualise_tsne_umap, "umap_arguments": umap_arguments,
                            "post_pca_scale": post_pca_scale, "scalers": scalers}
                
            # Write results to results folder along with the parameters dict
            results_and_params = {"results": results, "params": parameters_dict}

            # Save reults and parameters under dataset name and timestamp identifier
            timestamp = datetime.datetime.now()
            save_path = "results/{}_classical_{}_{}.json".format(filename, timestamp, neighbour)
            saved_location = write_dict(save_path, results_and_params)

            print("Results and parameters saved to {}".format(saved_location))

            # Set Pauli strings to create quantum feature maps
            pauli_strings = [['Z'], ['ZZ'], ['Z', 'ZZ'], ['Y', 'YY']]

            for pauli in pauli_strings:

                # Set pipeline variables for quantum kernels
                save_dim_info = True
                feature_map_str = pauli
                fm_arguments = {"reps": 2, "entanglement": "linear"}
                save_circuit = False
                sampler_options = {"optimization_level":3, "resilience_level":0,
                                    "execution":{"shots":1024, "seed":27} ,"simulator":{"precision":"single"}}
                fidelity_options = {"shots":1024, "seed":27}
                simulator = True
                qiskit_api_token = None
                qiskit_channel = 'ibm_quantum'
                backend_override = None
                min_qubits = 2
                sample_size = None
                classify_original = False

                # Run main function to obtain the results for the Pauli map based QSVM classifiers
                results = run_pipeline(filename=filename, dim_method=dim_method, minmax_val_list=minmax_val_list,
                                    n_dimensions=n_dimensions, save_dim_info=save_dim_info, tsne_arguments=tsne_arguments,
                                    visualise_tsne_umap=visualise_tsne_umap, umap_arguments=umap_arguments,
                                    post_pca_scale=post_pca_scale, scalers=scalers, feature_map_str=feature_map_str,
                                    fm_arguments=fm_arguments, save_circuit=save_circuit, sampler_options=sampler_options,
                                    fidelity_options=fidelity_options, simulator=simulator, qiskit_api_token=qiskit_api_token,
                                    qiskit_channel=qiskit_channel, backend_override=backend_override, min_qubits=min_qubits,
                                    sample_size=sample_size, classify_original=classify_original)
                print(results)

                 # Pass parameters to dictionary to record settings for pipeline run
                parameters_dict = {"filename": filename, "dim_method": dim_method, "minmax_val_list": minmax_val_list,
                                    "n_dimensions": n_dimensions, "save_dim_info": save_dim_info, "tsne_arguments": tsne_arguments,
                                    "visualise_tsne_umap": visualise_tsne_umap, "umap_arguments": umap_arguments,
                                    "post_pca_scale": post_pca_scale, "scalers": scalers, "feature_map_str": feature_map_str, 
                                    "fm_arguments": fm_arguments, "save_circuit": save_circuit, "sampler_options": sampler_options,
                                    "fidelity_options": fidelity_options, "simulator": simulator, "qiskit_api_token": qiskit_api_token,
                                    "qiskit_channel": qiskit_channel, "backend_override": backend_override, "min_qubits": min_qubits,
                                    "sample_size": sample_size, "classify_original": classify_original}
                
                # Write results to results folder along with the parameters dict
                results_and_params = {"results": results, "params": parameters_dict}

                # Save reults and parameters under dataset name and timestamp identifier
                timestamp = datetime.datetime.now()
                save_path = "results/{}_{}_pca.json".format(filename, timestamp)
                saved_location = write_dict(save_path, results_and_params)

                print("Results and parameters saved to {}".format(saved_location))
                