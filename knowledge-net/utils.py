import sys
from math import *
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from sklearn.model_selection import train_test_split
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
#import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split



# Functions allowed for data generation.
math_funcs = {"abs":abs, "sqrt":sqrt, "log":log, "log10":log10, "exp":exp} 


def generate_data(function,
                  noise_sd_func = 0,
                  data_size = 100, 
                  input_dim = 1,
                  lower = -10., 
                  upper = 10.):
    """ 
    Generate synthetic input and output data according to user-specified 
    functions.
    
    A data set of size <data_size> will be generated with <input_dim> i.i.d. 
    features drawn uniformly from [<lower>, <upper>]. Output will be generated 
    from input features according to <func> and <noise_sd>.

    Parameters
    ----------
    function : str
        String representation of the function to generate the output given the 
        inputs.
    noise_sd_func : str
        String representation of the function to generate the random noise of 
        the output. Default is 0.
    data_size : int
        Number of data points to generate. Default is 100.
    input_dim : int
        Number of input features. Default is 1.
    lower : float
        Lower bound on the input features domain. Default is -10.
    upper : float
        Upper bound on the input features domain. Default is 10.

    Returns
    -------
    X : numpy.ndarray
        Randomly-generated input data (bounded between lower and upper).
    y : numpy.ndarray
        Output (generated according to func and noise_sd_func).
    """
    
    # Set the function name.
    function_name = f"f(X) = {function} + E~N(0, {noise_sd_func})"                       
    
    # Initialize X and y tensors of correct shape.
    X = np.zeros(shape = (data_size, input_dim))
    y = np.zeros(shape = (data_size, 1))

    # Iterate for the desired data size to produce X and y vectors.
    for i in range(0, data_size):
        input_list = np.random.uniform(lower, upper, size = input_dim)
        
        try:
            # Generate output according to the user-specified function.
            math_funcs["x"] = input_list
            output = eval(function,{"__builtins__":None}, math_funcs)
            
            # Add in the noise generated from the specified noise standard
            # deviation.
            noise_sd = eval(noise_sd_func,{"__builtins__":None}, math_funcs)
            noise_sd=abs(noise_sd)
            noise = np.random.normal(0, noise_sd, 1)
            output += noise
            
        except IndexError:
            print("Ensure that the input_dim matches the dimensions of the "
                  "function.")
            sys.exit()

        X[i] = input_list
        y[i] = output

    return X, y



def create_dataset(X, y, test_size, batch_size):
    """
    Create a TensorFlow dataset from input data and split it into training and testing sets.

    This function takes feature data (X) and target data (y), performs a train-test split,
    and then converts the training data into a TensorFlow dataset. It also batches the
    training dataset for more efficient training.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        The feature data for the dataset.
    y : array-like or pd.Series
        The target data for the dataset.
    test_size : float
        The proportion of the dataset to include in the test split. It should be between
        0.0 and 1.0.
    batch_size : int
        The number of samples in each batch of the training dataset.

    Returns
    -------
    Tuple
        A tuple containing the following elements:
        - train_dataset : tf.data.Dataset
            A TensorFlow dataset containing the training data batched for efficient training.
        - test_dataset : tf.data.Dataset
            A TensorFlow dataset containing the testing data batched for efficient evaluation.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> train_dataset, test_dataset = create_dataset(
    ...         X, y, test_size=0.2, batch_size=32)
    """

    # Split the training and test data.
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
    
    # Load the datasets into a tf.data.Dataset object.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Batch the datasets.
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset




def load_mapping(mapping_file):
    """ 
    Input feature-id mapping file, return dictionary of feature-id mappings.

    Parameters
    ----------
    mapping_file : str
        Path to the file mapping features to ids.

    Returns
    -------
    mapping : dict
        Dictionary mapping features to their ids.
    """
    
    mapping_df = pd.read_csv(mapping_file, sep="\t").iloc[:, 0:2]
    mapping = dict(zip(mapping_df.feature, mapping_df.id))
    
    return mapping





def load_ontology(filename, feature_id_map):
    """
    Load the ontology file and return a directed acyclic graph to represent
    it.

    Parameters
    ----------
    filename : str
        Path to the ontology file.
    feature_id_map : dict
        Dictionary containing feature-id mapping.

    Returns
    -------
    G : nx.DiGraph
        Directed graph representing the ontology.
    root : str 
        The root node of the ontology.
    module_size_map : dict
        Dictionary containing total number of features annotated to a term 
        (directly or indirectly).
    module_direct_feature_map : dict
        Dictionary containing direct mapping between features and modules.
    """

    # Initialize an empty directed graph and dictionaries.
    G = nx.DiGraph()
    module_direct_feature_map = {}
    module_size_map = {}
    
    # Load the ontology file into a dataframe and iterate through rows.
    df = pd.read_csv(filename, sep="\t")
    for i, row in df.iterrows():
        parent = row["parent"]
        child = row["child"]
        relation = row["relation"]
        
        # Check that no disallowed characters exist.
        disallowed = [":", "/", "\\", ";", "_"]
        if any((char in parent) or (char in child) for char in disallowed):
            print(f"""
                  Line {i+2}, corresponding to the following  row in {filename} 
                  caused an error:
                  
                  {parent}\t{child}\t{relation} 
                  
                  Characters {', '.join(disallowed)} are not allowed in names.
                  """)
            sys.exit(1)
        
        # Ensure that feature names are lowercase.
        if relation != "module" and child.isupper():
            print(f"""
                  Line {i+2}, corresponding to the following  row in {filename} 
                  caused an error:
                  
                  {parent}\t{child}\t{relation} 
                  
                  All feature names must be lower-case.
                  """)
            sys.exit(1)
        
        # Ensure module names begin with uppercase letter.
        if (parent[0].islower()) or (relation == "module" and child[0].islower()):
            print(f"""
                  Line {i+2}, corresponding to the following  row in {filename} 
                  caused an error:
                  
                  {parent}\t{child}\t{relation} 
                  
                  All module names must begin with upper-case.
                  """)
            sys.exit(1)

        # If mapping between two modules, add to directed graph.
        if relation == "module":
            G.add_edge(parent, child)
        
        # If mapping between inputs and a module...
        else:
            if child not in feature_id_map:
                print(f"Feature {child} is not in the features.tsv file.")
                sys.exit(1)                    
            
            # If the module is mapped directly to inputs, instantiate a new 
            # set to record its inputs.
            if parent not in module_direct_feature_map:
                module_direct_feature_map[parent] = set()

            # Add the id of the input to the module that it is mapped to. 
            # Add this mapping to the directed graph.
            module_direct_feature_map[parent].add(feature_id_map[child])
            G.add_edge(parent, child)
    
    # Iterate through the modules in the directed graph.
    for module in G.nodes():
        
        # If the module is actually a feature, skip it.
        if not module[0].isupper():
            module_size_map[module] = 1
            continue
        
        # If the module has a direct mapping to an feature, 
        # add the feature to the module's feature set.
        module_feature_set = set()
        if module in module_direct_feature_map:
            module_feature_set = module_direct_feature_map[module]
        
        # Iterate through the descendents of the module.
        # Add any feature that are annotated to the child to the parents
        # feature set.
        deslist = nxadag.descendants(G, module)
        for des in deslist:                         
            if des in module_direct_feature_map:
                module_feature_set = module_feature_set | module_direct_feature_map[des]
        
        # If any of the terms have no features in their set, break.
        if len(module_feature_set) == 0:
            print("Module {module} is empty. Please delete it.")
            sys.exit(1)
        else:
            module_size_map[module] = len(module_feature_set)
    
    # Check that the ontology is fully connected and has only one root.
    leaves = [n for n,d in G.in_degree() if d==0]
    uG = G.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    if len(leaves) > 1:
        print(f"""
              There is more than 1 root of ontology. The roots are:

              {leaves}

              Please use only one root.
              """)
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print("There are more than connected components. Please connect them.")
        sys.exit(1)

    root = leaves[0]
    return G, root, module_size_map, module_direct_feature_map










def set_module_neurons(n, dynamic_neurons_func):
    """ Retrieve the number of neurons for a given module. 

    This function allows the user to specify the number of neurons a module 
    will contain as a function of the number of its inputs. It can be static, 
    where each module layer gets the same number of inputs, or it can be 
    dynamic, where the number of neurons is a function of the number of inputs 
    to the layer.
    """
   
    math_funcs["n"] = n
    mod_neurons = eval(dynamic_neurons_func, {"__builtins__":None}, math_funcs)
    
    return(mod_neurons)


















def build_meta_ontology(infile, outfile, module_file = None):
    meta_ont_df = pd.DataFrame()
    scores_df = pd.read_csv(infile)
    meta_ont_df["source_id"] = scores_df["Source"]
    meta_ont_df["source_name"] = scores_df["Source"]
    meta_ont_df["source_type"] = scores_df["Source"]
    meta_ont_df["target_id"] = scores_df["Target"]
    meta_ont_df["target_name"] = scores_df["Target"]
    meta_ont_df["target_type"] = scores_df["Target"]
    ind = scores_df.columns.get_loc("Target") + 1
    meta_ont_df["edge_value"] = scores_df.iloc[:, ind: ].mean(axis=1)
    
    
    source_ids_set = set(list(meta_ont_df["source_id"]))
    target_ids_set = set(list(meta_ont_df["target_id"]))
    root = list(target_ids_set.difference(source_ids_set))[0]
    ind = meta_ont_df.index[(meta_ont_df["target_id"] == root)][0]
    meta_ont_df.at[ind, "target_id"] = "Root"
    
    # Drop edges with an average score of 1.
    meta_ont_df = meta_ont_df.loc[~(meta_ont_df["edge_value"] == 1)]
    
    if module_file is not None:
        module_df = pd.read_csv(module_file)
        modules = list(module_df.iloc[:, 0])
        inputs = list(module_df.iloc[:, 1])
        inputs = [inp.lower() for inp in inputs]
        for mod, inp in zip(modules, inputs):
            ind = meta_ont_df.index[(meta_ont_df["source_name"] == inp)]
            if len(ind) != 0:
                ind = ind[0]
                meta_ont_df.at[ind, "source_type"] = mod 
    meta_ont_df.to_csv(outfile)









