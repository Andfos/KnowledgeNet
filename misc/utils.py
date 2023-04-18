import sys
from math import *
import numpy as np


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

                
