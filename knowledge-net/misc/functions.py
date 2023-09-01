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



math_funcs = {"abs":abs, "sqrt":sqrt, "log":log, "log10":log10, "exp":exp} 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "function", 
            help="Specify the function to model.",
            choices=["x**2", "sqrt(x)", "abEx", "x1*x2", "x1*x2+x3"])

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



# Generate a fully connected model to approximate the function x**2.
def x_squared_model(batch_size, initializer=initializers.GlorotUniform()):
    """ A 2-layer network for approximating x^2.

    Params
    ------
    batch_size : int
        The number of data points in a batch.
    initializer : keras.initializers.initializers
        The kernel initializer to use in the hidden layer.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        A neural network.
    """

    model = Sequential()
    model.add(Dense(
            2, activation="tanh", 
            use_bias=True, kernel_initializer=initializer))
    model.add(Dense(1, activation="linear", use_bias=True, name="Output"))
    model.compile(loss="mean_squared_error", optimizer="adam")  
    model.build(input_shape = (batch_size, 1))
    model.summary()
    
    return model



# Generate a fully connected model to approximate the function x1*x2.
def x1x2_model(batch_size, initializer=initializers.GlorotUniform()):
    
    """ A 2-layer network for approximating x^2.

    Params
    ------
    batch_size : int
        The number of data points in a batch.
    initializer : keras.initializers.initializers
        The kernel initializer to use in the hidden layer.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        A neural network.
    """

    model = Sequential()
    model.add(Dense(
            4, activation="tanh", 
            use_bias=True, kernel_initializer=initializer))
    model.add(Dense(1, activation="linear", use_bias=False, name="Output"))
    model.compile(loss="mean_squared_error", optimizer="adam")  
    model.build(input_shape = (batch_size, 2))
    model.summary()
    
    return model


if __name__ == "__main__":
    
    # Parse the command line arguments.
    args = parse_args()
    function = args.function
    size = int(args.size)
    lower = float(args.lower)
    upper = float(args.upper)
    noise = str(args.noise)
    
    # Loop through many random weight initializations.
    for seed in range(16, 100):
        initializer = initializers.GlorotUniform(seed = seed)

        if function == "x**2":
            func = function   
            input_dim = 1
            connections = np.ones((1, 1))
            model = x_squared_model(
                size,
                initializer=initializer)

        if function == "x1*x2":
            func = "x[0] * x[1]"
            input_dim = 2
            connections = np.ones((2, 2))
            connections = np.diag(np.diag(connections))
            model = x1x2_model(
                    connections=connections, 
                    initializer=initializer)
        
        # Generate the data according to the user specified function.
        X, y = generate_data(
                func,
                noise_sd_func=noise,
                data_size=size, 
                input_dim=input_dim,
                lower=lower, 
                upper=upper)

        # Fit the model and evaluate it.
        monitor_loss = EarlyStopping(
                monitor = "loss", patience = 1000, min_delta = 2)
        model.fit(X, y, epochs=10000, callbacks=[monitor_loss])
        rmse = round(sqrt(model.evaluate(X, y)), 2)

        # Plot the x**2 function approximation on a 2D plot.
        if function == "x**2" and rmse <= 5:
            
            # Generate plotting data.
            label = "Model: $L1_{2}^{tanh,bias}L2_{1}^{lin,bias}$"
            title = "Approximation of $x^{2}$"
            X_plot = np.linspace(2*lower, 2*upper, 1000)
            y_plot = X_plot**2
            preds = model.predict(X_plot)
            
            # Color left and right half of parabola differently.
            preds_left = preds.copy()
            preds_right = preds.copy()
            preds_left[X_plot >= 0] = np.nan
            preds_right[X_plot <= 0] = np.nan
            
            # Plot the predictions against the true function.
            plot_preds(
                    f"Images/x_squared/x_squared-{seed}", title, label,
                    X_plot, y_plot,
                    net_left = [preds_left, "$neuron_1$", "red"],
                    net_right = [preds_right, "$neuron_2$", "green"])


        # Plot the x1*x2 function approximation on a 3D plot.
        if function == "x1*x2" and rmse <= 5.:
            label = "$Model: L1_{4}^{tanh,bias}L2_1^{lin}$"
            
            # Generate 3D plot data
            x1, x2, X_plot = generate_3D_plot_data(
                    lower = -10, upper = 10, size = 100)

            y_plot = x1 * x2
            predicts = model.predict(X_plot).reshape(100, 100)
            
            # Generate a 3D plot.
            plot_preds_3D(
            f"Images/x1x2_{seed}",
            function, x1, x2, y_plot,
            title = "Approximation of $x_{1}*x_{2}$",
            net = [predicts, rmse, "blue", label])

