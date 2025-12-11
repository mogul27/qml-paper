import time
from qiskit.circuit.library import PauliFeatureMap
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import Sampler as QRSampler, Session, QiskitRuntimeService
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from utilities import get_optimal_backend, split_train_test_reduced
import pickle
import re

def create_pauli_kernels_sim(feature_map_str: list=['Z', 'ZZ'], fm_arguments: dict=None, 
                             save_circuit: bool=False, sampler_options: dict=None,
                             fidelity_options: dict=None, n_dimensions: tuple=(2,4),
                             classify_original: bool=False, original_dims: int=None):
    """
    Create quantum kernels to be used in classification that leverage a locally run
    quantum simulator

    Parameters:
        feature_map_str (str) -- Pauli string for Feature map to create. (default: ['Z', 'ZZ'])
        fm_arguments (dict) -- Optional arguments to creation of PauliFeatureMap (default: None)
        save_circuit (bool) -- Save quantum circuit layout drawing to fm_circuits folder (default: False)
        sampler_options (dict) -- Options to be passed to the creation of a Sampler instance (default: None)
        fidelity_options (dict) -- Options passed to ComputeUncompute fiedlity instance
        n_dimensions (tuple) -- Start/Stop value for kernel feature map dimensions (Inclusive) (default: 2, 4)
        classify_original (bool) -- Whether to classify original dataset (default: False)
        original_dims (int) -- Dimensionality of original dataset
    
    Returns:
        kernel_dict (dict) -- Dictionary with the kernel_name as keys and the constructed kernel as values
    """

    # Create sampler and fidelity for quantum kernels
    sampler = Sampler(options=sampler_options)
    fidelity = ComputeUncompute(sampler=sampler, options=fidelity_options)

    # Generate feature maps for each dimensionality required
    feature_maps = obtain_feature_maps(feature_map_str=feature_map_str, fm_arguments=fm_arguments,
                                       n_dimensions=n_dimensions, save_circuit=save_circuit,
                                       classify_original=classify_original, original_dims=original_dims)

    kernel_dict = {}

    # Create counter to use enumerate for iterating over dimensionality and loop
    # progress seperately
    dims = list(range(n_dimensions[0], n_dimensions[1]+1))

    # Feature map dict keys are in order of dimension so use indexing to retrieve correct
    # feature map for dimensionality
    feature_map_keys = list(feature_maps.keys())
    for count, dimensionality in enumerate(dims):
        
        # Extract feature map corresponding to dimensionality
        feature_map = feature_maps[feature_map_keys[count]]

        # Create kernels
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
          
        # Name kernel according to Pauli string and feature map dimensionality 
        kernel_name = "{}_kernel_{}d".format(feature_map_str, dimensionality)

        # Write kernels to dictionary
        kernel_dict[kernel_name]= kernel

    if classify_original:
        # Construct kernel from feature maps
        # Find feature map with original in key
        original_key = [key for key in feature_maps.keys() if re.search('original', key)][0]
        feature_map = feature_maps[original_key]
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        kernel_name = "{}_kernel_original_{}".format(feature_map_str, original_dims)
        kernel_dict[kernel_name] = kernel

    return kernel_dict

def evaluate_pauli_kernel_sim(kernel_dict: dict, sr_datasets: dict=None, n_dimensions:tuple=(2, 4),
                              classify_original: bool=False, original_dims: int=None):
    """
    Evaluate quantum kernels to create the train and test matrices used SVC
    classification on a quantum simulator
    
    Parameters:
        kernel dict (dict) -- Dictionary containing quantum kernels
        sr_datasets (dict) -- Dictionary containing scaled dimensionality reduced datasets for use in quantum kernel (default: None)
        n_dimensions (tuple) -- Start/Stop value for kernel feature map dimensions (Inclusive) (default: 2, 4)
        classify_original (bool) -- Whether to classify original dataset (default: False)
        original_dims (int) -- Dimensionality of original dataset
    
    Returns:
        computed_kernels (dict) -- Dictionary with kernel names as keys and computed test and train matrices as values
    """

    # Split all scaled and reduced datasets into train and test
    sr_train, sr_test = split_train_test_reduced(sr_datasets)

    # Retrieve keys as a list in ascending dimensionality order
    # to be indexed in the session loop
    sr_train_keys = list(sr_train.keys())
    sr_test_keys = list(sr_test.keys())

    # Create counter to access correct kernel for each dimension
    dims = list(range(n_dimensions[0], n_dimensions[1]+1))

    # Kernel dict keys are in order of dimension so use indexing to retrieve correct
    # kernel for dimensionality
    kernel_keys = list(kernel_dict.keys())

    # Create dictionary to hold kernel evaluation results
    computed_kernels = {}

    for count, dimensionality in enumerate(dims):

        # Obtain kernel, train and test set for given dimension
        kernel = kernel_dict[kernel_keys[count]]
        train_set = sr_train[sr_train_keys[count]]
        test_set = sr_test[sr_test_keys[count]]

        start = time.time()
        train_matrix = kernel.evaluate(x_vec=train_set)
        end = time.time()
        train_matrix_time = end - start

        # Store computed kernel and write somewhere in case of failure from
        # a later matrix run
        kernel_name = "{}_train".format(str(kernel_keys[count]))
        computed_kernels[kernel_name] = train_matrix
        save_path = "saved_sim_kernels/{}.pkl".format(kernel_name)
        with open(save_path, 'wb') as save_file:
            pickle.dump(computed_kernels[kernel_name], save_file)

        print("Total {}d train matrix evaluation time on quantum simulator: {}"
                  .format(dimensionality, train_matrix_time))
        
        start = time.time()
        test_matrix = kernel.evaluate(x_vec=test_set, y_vec=train_set)
        end = time.time()
        test_matrix_time = end - start

        # Store computed kernel and write somewhere in case of failure from
        # a later matrix run
        kernel_name = "{}_test".format(str(kernel_keys[count]))
        computed_kernels[kernel_name] = test_matrix
        save_path = "saved_sim_kernels/{}.pkl".format(kernel_name)
        with open(save_path, 'wb') as save_file:
            pickle.dump(computed_kernels[kernel_name], save_file)

        print("Total {}d test matrix evaluation time on quantum simulator: {}"
                  .format(dimensionality, test_matrix_time))
    
    if classify_original:
        # Get kernel for original dataset
        original_key = [key for key in kernel_dict.keys() if re.search('original', key)][0]
        kernel = kernel_dict[original_key]

        # Original train and test are always last items because they are appended
        train_set = sr_train[sr_train_keys[-1]]
        test_set = sr_test[sr_test_keys[-1]]
        
        start = time.time()
        train_matrix = kernel.evaluate(x_vec=train_set)
        end = time.time()
        train_matrix_time = end - start

        # Store computed kernel and write somewhere in case of failure from
        # a later matrix run
        kernel_name = "{}_train".format(str(original_key))
        computed_kernels[kernel_name] = train_matrix
        save_path = "saved_sim_kernels/{}.pkl".format(kernel_name)
        with open(save_path, 'wb') as save_file:
            pickle.dump(computed_kernels[kernel_name], save_file)

        print("Total {}d original train matrix evaluation time on quantum simulator: {}"
                  .format(original_dims, train_matrix_time))
        
        start = time.time()
        test_matrix = kernel.evaluate(x_vec=test_set, y_vec=train_set)
        end = time.time()
        test_matrix_time = end - start

        # Store computed kernel and write somewhere in case of failure from
        # a later matrix run
        kernel_name = "original_test_{}".format(str(original_key))
        computed_kernels[kernel_name] = test_matrix
        save_path = "saved_sim_kernels/{}.pkl".format(kernel_name)
        with open(save_path, 'wb') as save_file:
            pickle.dump(computed_kernels[kernel_name], save_file)

        print("Total {}d original test matrix evaluation time on quantum simulator: {}"
                  .format(original_dims, test_matrix_time))

    return computed_kernels

def evaluate_pauli_kernel_hw(feature_map_str: list=['Z', 'ZZ'], fm_arguments: dict=None, 
                             save_circuit: bool=False, sampler_options: dict=None,
                             fidelity_options: dict=None, n_dimensions: tuple=(2,4),
                             service: QiskitRuntimeService=None, backend_override: str=None,
                             sr_datasets: dict=None, min_qubits: int=2, classify_original: bool=False,
                             original_dims: int=None):
    
    """
    Create and evaluate Pauli Kernels for specified dimensionality on quantum hardware.

    Parameters: 
        feature_map_str (str) -- Pauli string for Feature map to create. (default: ['Z', 'ZZ'])
        fm_arguments (dict) -- Optional arguments to creation of PauliFeatureMap (default: None)
        save_circuit (bool) -- Save quantum circuit layout drawing to fm_circuits folder (default: False)
        sampler_options (dict) -- Options to be passed to the creation of a Qiskit Runtime Sampler
                                    instance (default: None)
        fidelity_options (dict) -- Options passed to ComputeUncompute fiedlity instance
        n_dimensions (tuple) -- Start/Stop value for kernel feature map dimensions (Inclusive) (default: 2, 4)
        service (QiskitRuntimeService) -- Instance of QiskitRuntimeService for using IBM quantum (default: None)
        backend_override (str) -- String representation of a backend to be used rather than the choseen
                                optimal backend (default: None)
        sr_datasets (dict): dict containing scaled dimensionality reduced datasets for use in quantum kernel (default: None)
        min_qubits (int) -- The minimum number of qubits in a quantum backend when selecting optimal backend (default: 2)
        classify_original (bool) -- Whether to classify original dataset (default: False)
        original_dims (int) -- Dimensionality of original dataset (default: None)

    Returns:
        computed_kernels (dict) -- Dictionary with kernel names as keys and computed test and train matrices as values
    """

    sampler_options = remove_sim_options(sampler_options)

    feature_maps = obtain_feature_maps(feature_map_str=feature_map_str, fm_arguments=fm_arguments,
                                        n_dimensions=n_dimensions, save_circuit=save_circuit)

    # Split all scaled and reduced datasets into train and test
    sr_train, sr_test = split_train_test_reduced(sr_datasets)

    # Retrieve keys as a list in ascending dimensionality order
    # to be indexed in the session loop
    sr_train_keys = list(sr_train.keys())
    sr_test_keys = list(sr_test.keys())

    if backend_override:
        backend = service.backend(backend_override)
    else:
        backend = get_optimal_backend(min_qubits=min_qubits, service=service)

    # Create counter to use enumerate for iterating over dimensionality and loop
    # progress seperately.
    dims = list(range(n_dimensions[0], n_dimensions[1]+1))

    # Feature map dict keys are in order of dimension so use indexing to retrieve correct
    # feature map for dimensionality
    feature_map_keys = list(feature_maps.keys())

    # Create dictionary to hold kernel results
    computed_kernels = {}

    for count, dimensionality in enumerate(dims):

        # Extract feature map according to dimensionality
        feature_map = feature_maps[feature_map_keys[count]]
        train_set = sr_train[sr_train_keys[count]]
        test_set = sr_test[sr_test_keys[count]]

        # Create session for computing kernel train matrix
        with Session(service=service, backend=backend, max_time='4h') as train_session:
        
            # Set up sampler and fidelity for quantum kernel
            sampler = QRSampler(session=train_session, options=sampler_options)
            fidelity = ComputeUncompute(sampler=sampler, options=fidelity_options)

            # Create kernel
            kernel = FidelityQuantumKernel(feature_map=feature_map,
                                        fidelity=fidelity)

            # Print out the time statistics and create train matrix
            print("Submitting train kernel training for dimensionality: {}".format(dimensionality))
            start = time.time()
            train_matrix = kernel.evaluate(x_vec=train_set)
            end = time.time()
            train_matrix_time = end - start

            # Store computed kernel and write somewhere in case failure from
            # a later session
            kernel_name = "{}_kernel_{}d_train".format(feature_map_str, dimensionality)
            computed_kernels[kernel_name] = train_matrix
            save_path = "saved_hw_kernels/{}.pkl".format(kernel_name)
            with open(save_path, 'wb') as save_file:
                pickle.dump(computed_kernels[kernel_name], save_file)

            print("Total {}d train matrix evaluation time on quantum hardware: {}"
                  .format(dimensionality, train_matrix_time))

            train_session.close()

        # Create session for computing kernel test_matrix
        with Session(service=service, backend=backend, max_time='4h') as test_session:
        
            # Set up sampler and fidelity for quantum kernel
            sampler = QRSampler(session=test_session, options=sampler_options)
            fidelity = ComputeUncompute(sampler=sampler, options=fidelity_options)

            # Create kernel
            kernel = FidelityQuantumKernel(feature_map=feature_map,
                                        fidelity=fidelity)
            
            # Print out the time statistics and create test matrix
            start = time.time()
            test_matrix = kernel.evaluate(x_vec=test_set, y_vec=train_set)
            end = time.time()
            test_matrix_time = end - start

            # Store computed kernel and write somewhere in case of failure from
            # a later session
            print("Submitting test kernel training for dimensionality: {}".format(dimensionality))
            kernel_name = "{}_kernel_{}d_test".format(feature_map_str, dimensionality)
            computed_kernels[kernel_name] = test_matrix
            save_path = "saved_hw_kernels/{}.pkl".format(kernel_name)
            with open(save_path, 'wb') as save_file:
                pickle.dump(computed_kernels[kernel_name], save_file)

            print("Total {}d test matrix evaluation time on quantum hardware: {}"
                  .format(dimensionality, test_matrix_time))
            
            test_session.close()

        if classify_original:
            original_key = [key for key in feature_maps.keys() if re.search('original', key)][0]
            feature_map = feature_maps[original_key]

            # Original train and test are always last items because they are appended
            train_set = sr_train[sr_train_keys[-1]]
            test_set = sr_test[sr_test_keys[-1]]

            # Create session for computing kernel train matrix
            with Session(service=service, backend=backend, max_time='4h') as train_session:
            
                # Set up sampler and fidelity for quantum kernel
                sampler = QRSampler(session=train_session, options=sampler_options)
                fidelity = ComputeUncompute(sampler=sampler, options=fidelity_options)

                # Create kernel
                kernel = FidelityQuantumKernel(feature_map=feature_map,
                                            fidelity=fidelity)

                # Print out the time statistics and create train matrix
                print("Submitting train kernel training for original dimensionality: {}".format(original_dims))
                start = time.time()
                train_matrix = kernel.evaluate(x_vec=train_set)
                end = time.time()
                train_matrix_time = end - start

                # Store computed kernel and write somewhere in case failure from
                # a later session
                kernel_name =  "{}_kernel_original_{}_train".format(feature_map_str, original_dims)
                computed_kernels[kernel_name] = train_matrix
                save_path = "saved_hw_kernels/{}.pkl".format(kernel_name)
                with open(save_path, 'wb') as save_file:
                    pickle.dump(computed_kernels[kernel_name], save_file)

                print("Total original {}d train matrix evaluation time on quantum hardware: {}"
                    .format(original_dims, train_matrix_time))

                train_session.close()

        # Create session for computing kernel test_matrix
        with Session(service=service, backend=backend, max_time='4h') as test_session:
        
            # Set up sampler and fidelity for quantum kernel
            sampler = QRSampler(session=test_session, options=sampler_options)
            fidelity = ComputeUncompute(sampler=sampler, options=fidelity_options)

            # Create kernel
            kernel = FidelityQuantumKernel(feature_map=feature_map,
                                        fidelity=fidelity)
            
            # Print out the time statistics and create test matrix
            start = time.time()
            test_matrix = kernel.evaluate(x_vec=test_set, y_vec=train_set)
            end = time.time()
            test_matrix_time = end - start

            # Store computed kernel and write somewhere in case failure from
            # a later session
            print("Submitting test kernel training for original dimensionality: {}".format(original_dims))
            kernel_name = "{}_kernel_original_{}_test".format(feature_map_str, original_dims)
            computed_kernels[kernel_name] = test_matrix
            save_path = "saved_hw_kernels/{}.pkl".format(kernel_name)
            with open(save_path, 'wb') as save_file:
                pickle.dump(computed_kernels[kernel_name], save_file)

            print("Total original {}d test matrix evaluation time on quantum hardware: {}"
                  .format(original_dims, test_matrix_time))
            
            test_session.close()

    return computed_kernels

def remove_sim_options(sampler_options: dict):
    """
    Check if simulator options or simulator seed are set for sampler using hardware and 
    if so, remove them.

    Parameters:
        sampler_options (dict) -- Options to be modified before being passed to the creation of a Qiskit
                              Runtime Sampler instance
    Returns:
        sampler_options (dict) -- Sampler options with simulator related items removed
    """

    # Remove full simulaltor dictionary
    sampler_options.pop('simulator' , None)

    # Remove seed options from execution
    if 'execution' in sampler_options.keys():
        sampler_options['execution'].pop('seed', None)

    return sampler_options
    

def obtain_feature_maps(feature_map_str: list=['Z', 'ZZ'], fm_arguments: dict=None, 
                        n_dimensions: tuple=(2,4), save_circuit: bool=False,
                        classify_original: bool=False, original_dims: int=None):
    """
    Create a set of feature maps for the dimensionality and feature map specified.

    Parameters:
        feature_map_str (str) -- Pauli string for Feature map to create. (default: ['Z', 'ZZ'])
        fm_arguments (dict) -- Optional arguments to creation of PauliFeatureMap (default: None)
        n_dimensions (tuple) -- Start/Stop value for feature map dimensions (Inclusive) (default: 2, 4)
        save_circuit (bool) -- Save quantum circuit layout drawing to fm_circuits folder (default: False)
    classify_original (bool) -- Whether to classify original dataset (default: False)
        original_dims (int) -- Dimensionality of original dataset

    Returns:
        feature_maps (dict) -- Dictionary containing feature maps for kernel creation
    """
    
    feature_maps = {}

    # If optional arguments passed, set those for PauliFeatureMap creation
    if fm_arguments:
        reps=fm_arguments['reps']
        entanglement=fm_arguments['entanglement']

    # Otherwise explicitly set to default values to those in qiskit documentation
    else:
        reps=2
        entanglement='full'

    for dimensionality in range(n_dimensions[0], n_dimensions[1]+1):
        
        # Create Feature Maps
        pauli_fm = PauliFeatureMap(feature_dimension=dimensionality, reps=reps,
                                   entanglement=entanglement, paulis=feature_map_str)
          
        # Name feature map according to Pauli string and dimensionality
        fm_name = "{}_pauli_{}d".format(feature_map_str, dimensionality)

        # save feature maps
        feature_maps[fm_name]= pauli_fm

        if save_circuit:
            pauli_fm.decompose().draw(output='mpl',
                                    filename="fm_circuits/{}.png".format(fm_name))
    
    # if classify_original then also create feature map for original dataset
    if classify_original:
        pauli_fm = PauliFeatureMap(feature_dimension=original_dims, reps=reps,
                                   entanglement=entanglement, paulis=feature_map_str)
        # Name feature map according to Pauli string
        fm_name = "{}_pauli_original_{}".format(feature_map_str, original_dims)

        # save feature maps
        feature_maps[fm_name]= pauli_fm

        if save_circuit:
            pauli_fm.decompose().draw(output='mpl',
                                    filename="fm_circuits/{}.png".format(fm_name))

    return feature_maps