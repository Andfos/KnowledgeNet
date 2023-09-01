import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import statistics as stats



def plot_preds(filepath, title, label, X, y, **kwargs):
    """ Plots predicted values against underlying function.

    Params
    ------
    filepath : str
        Path to save image file.
    title : str
        Title of the plot.
    label : str
        Label for the plot (shown in legend).
    X : numpy.ndarray
        The input features to the function.
    y : numpy.ndarray
        The noiseless output values of the function.
    **kwargs : {}
        Used to plot additional lines for predictions. Each key in kwargs 
        refers to a new line to plot. Each key is mapped to a list with 
        the following structure: [y_predictions, legend_label, color].
    """

    # Set the axes of the figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the underlying function.
    plt.plot(X, y, 'b')
    
    # Plot the predictions.
    for key, value in kwargs.items():
        plt.plot(X, value[0], value[2], label = f"{value[1]}")
    plt.title(title)
    ax.legend(
            loc = "upper right", bbox_to_anchor=(0.9, 1), 
            title = label, fancybox=True, framealpha=0.2)
    plt.savefig(filepath)




def plot_loss(filepath, title, **kwargs): 
    
    """ Function to plot the traiing versus validation loss"""
    fig, ax = plt.subplots()
    
    # Plot the loss curves
    plt.figure(figsize=(12.8,4.8))
    for key, value in kwargs.items():
        plt.plot(
            value[0], 
            color = value[1], 
            label = f"{values[1]}")

    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.savefig(filepath)
    plt.close()





def plot_preds_3D(filepath, function, x, y, z, title, **kwargs):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(x, y, z, color='black')
    for key, value in kwargs.items():
        ax.plot_wireframe(x, y, value[0], 
                          color = value[2], 
                          label = "{}: {} rmse".format(value[3], value[1]))
    # Set the axes
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title(title)
    plt.legend()
    fig.tight_layout()
    fig.subplots_adjust(right=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1.08, 0.93), fontsize=7)
    plt.savefig(filepath)
    plt.show()
    




def generate_3D_plot_data(lower = -10, upper = 10, size = 100):
    """ This function returns the input data required for making 3D
    plots."""
    # Generate evenly spaced x1 and x2 values; create a meshgrid
    x1 = np.linspace(lower, upper, size)
    x2 = np.linspace(lower, upper, size)
    x1, x2 = np.meshgrid(x1, x2)

    # Generate model input data
    X_modelInp = np.zeros(shape = (size*size,2))
    k = 0
    for i in range(len(x1)):
        for j in range(len(x2)):
            arr = np.array([x1[0][i], x2[j][0]])
            X_modelInp[k] = arr
            k += 1
    
    return x1, x2, X_modelInp



