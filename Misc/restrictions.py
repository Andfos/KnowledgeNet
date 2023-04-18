import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from plots import *
from utils import *
from math import *
import keras.backend as K







def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-s", "--size",
            help="Specify the number of data points to generate (default = 1000).", 
            default=1000)

    parser.add_argument(
            "-l", "--lower",
            help="Specify the lower bound of the data to generate (default = -10).", 
            default=-10)

    parser.add_argument(
            "-u", "--upper",
            help="Specify the upper bound of the data to generate (default = 10).",
            default=10)

    parser.add_argument(
            "-n", "--noise", 
            help="""Specify the function to generate the standard deviation of the
                    noise of the data (default = 0). The mean of the noise is 
                    always centered at 0, and it's standard deviation can be 
                    specified by a single number (e.g. 3), or by a function of x 
                    (e.g. sqrt(x)).""",
            default=0)

    args = parser.parse_args()
    return args




class RestrictedLayer(Dense):
    """ Build a layer where a specific connections matrix can be applied.  """
    
    def __init__(self, units, connections, pbias=None, **kwargs):
        """ 
        Initialize a RestrictedLayer that inherits from the Keras Dense
        class.

        Args:
            units(int): Number of nodes for layer
            connections(numpy.ndarray): Connections mask for weights 
                    leaving current layer.
            pbias(float): Bias to be added to the final output of the node 
                    following the activation.
        """

        super().__init__(units, **kwargs)
        
        self.connections = connections
        
        if pbias is not None:
            self.pbias = tf.Variable([pbias], name=f"{self.name}/pbias")

    def call(self, inputs):
        """ 
        Call the restricted layer on a set of inputs.

        Operation: output = activation(dot(input, kernel)+ bias) + pbias

        Args:
            inputs

        """
        output = K.dot(inputs, self.kernel * self.connections)
        

        # If the bias is to be included, add it with the output of the previous
        # step.
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        
        # Apply the activation function and then add the pbias term.
        output = self.activation(output)
        
        if hasattr(self, "pbias"):
            output = tf.math.add(self.pbias, output)
        
        return output















def x1x2_plus_x3_model(
            size,
            input_dim,
            connections,
            initializer=initializers.GlorotUniform()):
    model = Sequential()
    model.add(RestrictedLayer(
            5, connections, pbias=None, activation="tanh", use_bias=True,
            kernel_initializer=initializer))
    model.add(Dense(1, activation="linear", use_bias=False, name="Output"))
    model.compile(loss="mean_squared_error", optimizer="adam")  
    model.build(input_shape = (size, input_dim))
    model.summary()
    
    return model





if __name__ == "__main__":



    # Parse the command line arguments.
    args = parse_args()
    size = int(args.size)
    lower = float(args.lower)
    upper = float(args.upper)
    noise = str(args.noise)
    
    
    func = "x[0] * x[1] + x[2]"
    input_dim = 3
    dense_connections = np.ones((3, 5))
    res_connections = np.ones((3, 5))
    res_connections[-1] = 0
    res_connections[:, -1] = 0
    res_connections[-1, -1] = 1
    

    # Generate the data according to the user specified function.
    X, y = generate_data(
            func,
            noise_sd_func=noise,
            data_size=size, 
            input_dim=input_dim,
            lower=lower, 
            upper=upper)


    # Loop through many random weight initializations.
    for seed in range(0, 100):
        print("seed {seed}".format(seed = seed))        
        initializer = initializers.GlorotUniform(seed = seed)

        dense_model = x1x2_plus_x3_model(
            size,
            input_dim,
            connections=dense_connections,
            initializer=initializer)
        res_model = x1x2_plus_x3_model(
            size,
            input_dim,
            connections=res_connections,
            initializer=initializer)

        
        # Fit the model and evaluate it.
        monitor_loss = EarlyStopping(
                monitor = "loss", patience = 2500, min_delta = 2)
        dense_history = dense_model.fit(X, y, epochs=10000, callbacks=[monitor_loss])
        res_history = res_model.fit(X, y, epochs=10000, callbacks=[monitor_loss])
        dense_rmse = round(sqrt(dense_model.evaluate(X, y)), 2)
        res_rmse = round(sqrt(res_model.evaluate(X, y)), 2)




        # Generate plotting data.
        dense_label = "Fully-connected Model: $L1_{5}^{tanh,bias}L2_{1}^{lin}$"
        res_label = "Constrained Model: $L1_{5}^{tanh,bias}L2_{1}^{lin}$"
        title = "Approximation of $x_{1} \cdot x_{2} + x_{3}$"
        
        if dense_rmse <= 100 or res_rmse <= 100:
            plot_loss(
                    f"Images/x1x2_x3/x1x2_x3-{seed}",
                    title,
                    Dense = [dense_history.history["loss"],
                            "red",
                            dense_label,
                            dense_rmse],
                    Res = [res_history.history["loss"],
                            "blue",
                            res_label,
                            res_rmse])

        

