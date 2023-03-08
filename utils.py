import sys
from math import *
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Functions allowed for data generation.
math_funcs = {"abs":abs, "sqrt":sqrt, "log":log, "log10":log10, "exp":exp} 


def generate_data(function,
                  noise_sd_func = 0,
                  data_size = 100, 
                  input_dim = 1,
                  lower = -10., 
                  upper = 10.):
    """ 
    Manually generate the input and output data according to user-specified 
    functions.

    Args:
        function(str): String representation of the function to generate the
                output given the inputs
        noise_sd_func(str): String representation of the function to generate 
                the random noise of the output.
        data_size(int): Number of data points to generate (default 100)
        input_dim(int): Number of input features (default 1)
        lower(float): Lower bound on the input features domain.
        upper(float): Upper bound on the input features domain.

    Returns:
        X(numpy.ndarray): Input features (bounded between lower and upper)
        y(numpy.ndarray): Output (generated according to func and noise_sd_func)
    """
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







def set_module_neurons(n, dynamic_neurons_func):
    """ Retrieve the number of neurons for a given module. 

    This function allows the user to specify the number of neurons a module 
    will contain as a function of the number of its inputs.
    """
    math_funcs["n"] = n
    mod_neurons = eval(dynamic_neurons_func, {"__builtins__":None}, math_funcs)
    return(mod_neurons)






def load_mapping(mapping_file):
    """ 
    Input gene-id mapping file, return dictionary of gene-id mappings.

    Args:
        mapping_file(str): Path to the file mapping inputs to ids.

    Returns:
        mapping(dict): Dictionary mapping inputs to their ids.
    """
    
    mapping = {}
    file_handle = open(mapping_file)
    
    for line in file_handle:
        line = line.rstrip().split()
        mapping[line[1]] = int(line[0])

    file_handle.close()
    
    return mapping






def load_ontology(file_name, input_id_map):
    """
    Load the ontology file and return a directed acyclic graph to represent
    it.
    

    Args:
        file_name(str): Path to the ontology file
        input_id_map(dict): Dictionary containing input-id mapping

    Returns:
        G(nx.DiGraph): Directed graph representing the ontology
        root(str): The root node of the ontology
        module_size_map(dict): Dictionary containing total number of inputs 
                annotated to a term (directly or indirectly)
        module_direct_input_map(dict): Dictionary containing direct mapping 
                between inputs and modules.
    """

    # Initialize an empty directed graph and sets
    G = nx.DiGraph()
    module_direct_input_map = {}
    module_size_map = {}

    # Iterate through the ontology file.
    file_handle = open(file_name)
    for i, fline in enumerate(file_handle):
        line = fline.rstrip().split()
        parent = line[0]
        child = line[1]
        relation = line[2]
        
        # Check that names do not have special characters and inputs are
        # lowercase.
        disallowed = [":", "/", "\\", ";", "_"]
        if any((char in parent) or (char in child) for char in disallowed):
            print(f"Line {i} in {file_name} caused an error.\n\n{fline}")
            print(f"Characters {', '.join(disallowed)} not allowed in names.")
            sys.exit(1)
        if relation != "default" and child[0].isupper():
            print(f"Line {i} in {file_name} caused an error.\n\n{fline}")
            print(f"{child} is an input and should be lowercase.")
            sys.exit(1)
        

        # If mapping between two modules, add to directed graph.
        if relation == 'default':
            G.add_edge(parent, child)
        
        # If mapping between inputs and a module...
        else:
            if child not in input_id_map:
                print("Input {child} not in the input id map file")
                sys.exit(1)                    
            
            # If the module is mapped directly to inputs, instantiate a new 
            # set to record its inputs.
            if parent not in module_direct_input_map:
                module_direct_input_map[parent] = set()

            # Add the id of the input to the module that it is mapped to. 
            # Add this mapping to the directed graph.
            module_direct_input_map[parent].add(input_id_map[child])
            G.add_edge(parent, child)

    file_handle.close()

    # Iterate through the modules in the directed graph.
    for module in G.nodes():
        
        # If the module is an input, skip it.
        if not module[0].isupper():
            module_size_map[module] = 1
            continue

        # If the module has a direct mapping to an input, 
        # add the input to the module's input set.
        module_input_set = set()
        if module in module_direct_input_map:
            module_input_set = module_direct_input_map[module]
        
        # Iterate through the descendents of the module.
        # Add any inputs that are annotated to the child to the parents
        # input set.
        deslist = nxadag.descendants(G, module)
        for des in deslist:                         
            if des in module_direct_input_map:
                module_input_set = module_input_set | module_direct_input_map[des]
        
        # If any of the terms have no inputs in their set, break.
        if len(module_input_set) == 0:
            print("Module {module} is empty. Please delete it.")
            sys.exit(1)
        else:
            module_size_map[module] = len(module_input_set)
    
    # Check that the ontology is fully connected and has only one root.
    leaves = [n for n,d in G.in_degree() if d==0]
    uG = G.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print('There are more than connected components. Please connect them.')
        sys.exit(1)


    root = leaves[0]
    return G, root, module_size_map, module_direct_input_map







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









def debugger(mode, run_debugger = True, **kwargs):

    if mode == "input_layer" and run_debugger == True:
        var_name = kwargs["var_name"]
        init_val = kwargs["init_val"]
        lr = kwargs["lr"]
        dW = kwargs["dW"]
        temp_val = kwargs["temp_val"]

        print(f"\nUpdating {var_name}...\n")
        new_var = init_val - lr*dW
        print("Initial weights - (lr * gradient) = New weights")
        for b in range(len(init_val)):
            print(f"{init_val[b]}\t-\t{lr} * {dW[b]}\t=\t{temp_val[b]}\n")
        print("\n")            

    # Debugging for proximal L0 (penalties.py)
    if mode == "L0" and run_debugger == True:
        alpha_abs = kwargs["alpha_abs"]
        theta = kwargs["theta"]
        alpha_new = kwargs["alpha_new"]

        alpha_array = np.array(alpha_abs.numpy())
        print("Abs(Weights) <> Theta = New weights")
        for b in range(len(alpha_array)):
            print(f"{alpha_abs[b]}\t>\t{theta}\t=\t{alpha_new[b]}\n")
        print("\n\n")

    # Debugging for weight updates between the input layer and a module layer
    # directly mapped to an input layer.
    if mode == "module_layer" and run_debugger == True:
        child_name = kwargs["child_name"]
        mod_name = kwargs["mod_name"]
        init_weights = kwargs["init_weights"]
        lr = kwargs["lr"]
        dW = kwargs["dW"]
        temp_weights = kwargs["temp_weights"]


        
        print(f"\nUpdating weights from {child_name} that feed ", 
              f"into {mod_name}...\n")
        
        print("Initial weights - (lr * gradient) = New weights")
        
        if init_weights.ndim > 1:
            for a in range(len(init_weights)):
                print(f"{init_weights[a]}\t-\t{lr} * {dW[a]}\t=\t{temp_weights[a]}")
            print("\n")            
        
        else:
            print(f"{init_weights}\t-\t{lr} * {dW}\t=\t{temp_weights}")
            print("\n")            
    
    if mode == "group_lasso" and run_debugger == True:
        alpha = kwargs["alpha"]
        alpha_norm = kwargs["alpha_norm"]
        c = kwargs["c"]
        alpha_new = kwargs["alpha_new"]

    
        print("If matrix norm < c, matrix --> 0")
        print("Otherwise, matrix --> (matrix / matrix_norm) * (matrix_norm - c)")

        print(f"Matrix norm: {alpha_norm} <> C: {c}")

        if alpha.ndim > 1:
            for b in range(len(alpha)):
                print(f"{alpha[b]}\t-->\t{alpha_new[b]}")
            print("\n\n")
        else:
            print(f"{alpha}\t-->\t{alpha_new}")
            print("\n\n")






