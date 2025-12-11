from qiskit_ibm_runtime import QiskitRuntimeService
import json

def qiskit_login(token:str=None, channel: str='ibm_quantum'):
    """
    Login to Qiskit Runtime for access to quantum hardware and simulators from
    IBM Quantum. If token is None then credentials will be loaded from disk. Otherwise,
    if a token is supplied then the token will be used for a one-time-login."

    Parameters:
        token (str) -- API Login token from IBM Quantum (default: None)
        channel (str) -- Type of account to use to access IBM Quantum (default: 'ibm_quantum')

    Returns:
        service (QiskitRuntimeService) -- Instance of QiskitRuntimeService for using IBM quantum
    """

    if token:
        service = QiskitRuntimeService(channel=channel,
                               token=token)
        return service
        
    else:
        print("No token provided, using saved credentials")
        service = QiskitRuntimeService()
        print("Login successful. Qiskit Runtime Service initialised")

        return service

def get_optimal_backend(min_qubits: int=2, service: QiskitRuntimeService=None):
    """Return the least busy quantum hardware backend that meets qubit requirements.
    
    Parameters:
        min_qubits (int) -- The minimum number of qubits in a backend (default: 2)
        service (QiskitRuntimeService) -- Instance of QiskitRuntimeService for using IBM quantum
    
    Returns:
        optimal_backend (IBMBackend) -- Quantum device or simulator hosted on IBM Quantum

    """
    
    optimal_backend = service.least_busy(min_num_qubits=min_qubits, operational=True,
                                         simulator=False)

    print("Using backend: {}".format(optimal_backend))
    
    return optimal_backend

def split_train_test_reduced(sr_datasets: dict):
    """
    Split the dictionary containing all train and test datasets into seperate
    dictionaries containing only train or only test datasets

    Parameters:
        sr_datasets (dict) -- A dictionary containing all scaled and reduced datasets

    Returns:
        train_dict (dict) -- A dictionary containing all scaled and reduced train sets
        test_dict (dict) -- A dictionary containing all scaled and reduced test sets
    """

    train_dict = {}
    test_dict = {}

    for key, value in sr_datasets.items():
        if "train" in key:
            train_dict[key] = value
        else:
            test_dict[key] = value

    return train_dict, test_dict

def split_train_test_kernels(evaluated_kernels: dict):
    """
    Split the dictionary containing all train and test kernels into seperate
    dictionaries containing only train or only test kernels

    Parameters:
        evaluated_kernels (dict) -- A dictionary containing all evaluated kernels

    Returns:
        train_dict (dict) -- A dictionary containing all evaluated train kernels
        test_dict (dict) -- A dictionary containing all evaulated test kernels
    """

    train_dict = {}
    test_dict = {}

    for key, value in evaluated_kernels.items():
        if "train" in key:
            train_dict[key] = value
        else:
            test_dict[key] = value

    return train_dict, test_dict

def write_dict(save_path: str, save_dict: dict):
    """
    Save a given dictionary as a json object

    Parameters:
        save_path (dict): Path to directory in which to save the json file
        save_dict (dict): Dictionary to store as json object

    Returns:
        save_path (dict): Path to directory in which the json file is saved
    """

    with open(save_path, 'w') as save_file:
        json.dump(save_dict, save_file, indent=4)

    return save_path
    









