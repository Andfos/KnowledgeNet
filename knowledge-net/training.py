import tensorflow as tf
import numpy as np
import networkx as nx
from utils import debugger
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import re

np.set_printoptions(suppress=True)



def sparse_group_lasso(w, lamb1, eta1, lamb2, eta2):
    """ Applies the sparse group lasso on a kernel of weights.

    Parameters
    ----------
    w : 
    lamb1 : float
        Group penalty term when group is defined as weights incoming to a node.
    eta1 : float
        The l0 penalty term applied to individual weights within a group, when 
        a group is defined as the weights incoming to a node.
    lamb2 : float
        Group penalty term when group is defined as weights outgoing from a node.
    eta2 : float
        The l0 penalty term applied to individual weights within a group, when 
        a group is defined as the weights outgoing from a node.
    """
    
    w = w.numpy()
    S = tf.zeros_like(w).numpy()       
    lamb = max(lamb1, lamb2)

    # For updating biases.
    if len(w.shape)<2:
        S = tf.where(tf.math.abs(w) > lamb1, 0.5 * (w - lamb1)**2 * eta1, S)
        w_new = tf.where(S > 0, (1- lamb1/tf.math.abs(w)) * w, tf.zeros_like(w))
    
    # For updating weights.
    else:
        
        # Induce sparsity within columns.
        if eta1 != 0.0:
            #print("sparsity by column")
            w_ones = tf.ones_like(w).numpy()
            w_zeros = tf.zeros_like(w).numpy()
            w_tens = w_ones * 10
            
            # Record the column indices of each element in w.
            c_diag = tf.cast(tf.linalg.diag(tf.range(0, w.shape[1])), tf.float32)
            cols = tf.matmul(w_ones,c_diag)
            
            # Sort w by column, such that the maximum of each column is in the 
            # 1st row.
            sorted_w = tf.sort(tf.math.abs(w), axis=0, direction='DESCENDING')
            
            # Record the original row indices of each element in sorted_w.
            indices = tf.argsort(tf.math.abs(w), axis=0, direction='DESCENDING')
            
            # Generate y_i, which should get larger as the row index increases.
            lower_tri = tf.linalg.band_part(
                    tf.ones((w.shape[0], w.shape[0])), -1, 0)
            y_i = tf.math.sqrt(tf.matmul(lower_tri, sorted_w**2))
            
            # Create a matrix representing the relative penalty applied to deeper
            # rows (rows further down in sorted_w).
            tri = tf.cast(tf.linalg.diag(tf.range(1, w.shape[0]+1)),tf.float32)
            row_penalty = tf.matmul(tri, w_tens)
            
            # See if an entire column can be zeroed-out. Alternatively apply a 
            # penalty that can be used to induce individual column sparsity.
            S = tf.where(y_i > lamb1,
                         0.5*(y_i - lamb1)**2 - eta1*row_penalty,
                         S)
            
            # Obtain the maximum row index of each column of sorted_w that should 
            # not be zerod out.
            values = tf.math.reduce_max(S, axis=0)
            row_ind = tf.math.argmax(S, 0)
            row_indx = tf.where(values>0, row_ind+1, tf.zeros_like(row_ind))
            
            # Create a sparsity mask.
            ind0 = tf.where(tf.matmul(w_ones, tf.cast(tf.linalg.diag(row_indx),
                                                      tf.float32))
                            < tf.matmul(tri, w_ones),
                            w_zeros, w_ones)
            
            # Retrieve the orginal elements of w, or set the element to 0.
            w_zeros[indices, tf.cast(cols, tf.int64)] = sorted_w * ind0
            w = w_zeros*tf.math.sign(w)
                                                                        
        

        # Induce sparsity within rows.
        if eta2 != 0.0:
            #print("sparsity by row")
            S = tf.zeros_like(w).numpy()       
            w_zeros = tf.zeros_like(w).numpy()
            w_ones = tf.ones_like(w).numpy()
            
            # Record the row indices of each element in w.
            c_diag = tf.cast(tf.linalg.diag(tf.range(0, w.shape[0])), tf.float32)
            rows = tf.matmul(c_diag, w_ones)
            
            # Sort w by row, such that the maximum of each row is in the 1st column.
            sorted_w = tf.sort(tf.math.abs(w), axis=1, direction='DESCENDING')
            
            # Record the original column indices of each element in sorted_w.
            indices = tf.argsort(tf.math.abs(w), axis=1, direction='DESCENDING')
            
            # Generate y_i, which should get larger as the column index increases.
            upper_tri = tf.linalg.band_part(
                    tf.ones((w.shape[1], w.shape[1])), 0, -1)
            y_i = tf.math.sqrt(tf.matmul(sorted_w**2, upper_tri))
            
            # Create a matrix representing the relative penalty applied to deeper
            # columns (columns further to the right in sorted_w).
            tri = tf.cast(tf.linalg.diag(tf.range(1, w.shape[1]+1)),tf.float32)
            column_penalty = tf.matmul(w_ones, tri)
            
            
            # See if an entire row can be zeroed-out. Alternatively apply a penalty 
            # that can be used to induce individual row sparsity.
            S = tf.where(y_i > lamb,
                         0.5*(y_i - lamb)**2 - eta2*column_penalty,
                         S)

            # Obtain the maximum column index of each row of sorted_w that should 
            # not be zerod out.
            values = tf.math.reduce_max(S, axis=1)
            col_ind = tf.math.argmax(S, 1)

            # Get the maximum column index to keep for each row in sorted_w, or
            # keep no columns and set the entire row to 0.
            col_indx = tf.where(values>0, col_ind+1, tf.zeros_like(col_ind))
            
            # Create a sparsity mask.
            ind0 = tf.where(tf.matmul(tf.cast(tf.linalg.diag(col_indx), tf.float32), 
                                     w_ones) < tf.matmul(w_ones, tri),
                            w_zeros, w_ones)
            
            # Retrieve the orginal elements of w, or set the element to 0.
            w_zeros[tf.cast(rows, tf.int64), indices] = sorted_w * ind0
            w = w_zeros*tf.math.sign(w)

        w_new = w

        # Normalization by column.
        if lamb1 != 0.0:
            #print("normalizing by column")
            col_norm = tf.norm(w_new, ord="euclidean", axis=0)
            norm_coef1 = tf.where(
                    col_norm == 0, tf.zeros_like(col_norm), 1 - lamb1/col_norm)
            w_new = w_new * norm_coef1
        
        
        # Normalization by row.
        if lamb2 != 0.0:
            #print("normalizing by row")
            row_norm = tf.norm(w_new, ord="euclidean", axis=1)
            norm_coef2 = tf.where(
                    row_norm == 0, tf.zeros_like(row_norm), 1 - lamb2/row_norm)
            #w_temp = tf.zeros_like(w_new).numpy()
            w_new = w_new.numpy()
            for index in range(len(norm_coef2)):
                w_new[index] = w_new[index] * norm_coef2[index]
    

    return w_new







def get_loss(model, y_true, y_pred, reg_penalty=True):
    """
    Calculate the loss for a model's prediction given the true labels.

    This function calculates the loss between the model's predicted values and the
    true labels using the model's specified loss function. Optionally, it can add
    regularization loss to the computed loss.

    Parameters
    ----------
    model : tf.keras.Model
        The model for which the loss is calculated. It should be a TensorFlow Keras
        Model object.

    y_true : tf.Tensor
        The true labels or ground truth values corresponding to the model's
        predictions. It should be a TensorFlow Tensor object.

    y_pred : tf.Tensor
        The predicted values generated by the model. It should be a TensorFlow Tensor
        object of the same shape as `y_true`.

    reg_penalty : bool, optional
        A flag indicating whether to include regularization loss in the computed loss.
        If True, the regularization loss from the model's internal `losses` property
        will be added to the final loss. Default is True.

    Returns
    -------
    tf.Tensor
        The computed loss value as a TensorFlow Tensor object.
    """
    
    if model.output_act == "softmax":
        pass
    y_pred = tf.cast(y_pred, tf.float32)
    loss = model.loss_fn(y_true=y_true, y_pred=y_pred)
    
    # Add in the regularization loss.
    if reg_penalty:
        loss += sum(model.losses)

    return loss




def get_accuracy(model, truth, preds):
    """
    Calculate the accuracy of a model's predictions compared to the true labels.

    This function calculates the accuracy of the model's predicted values compared to
    the true labels. It is used to evaluate the performance of a model in a
    classification task.

    Parameters
    ----------
    model : tf.keras.Model
        The model for which the accuracy is calculated. It should be a TensorFlow Keras
        Model object.

    truth : np.ndarray or list
        The true labels or ground truth values corresponding to the model's
        predictions. It should be a numpy array or list.

    preds : tf.Tensor
        The predicted values generated by the model. It should be a TensorFlow Tensor
        object of the same shape as `truth`.

    Returns
    -------
    float
        The accuracy of the model's predictions as a floating-point number between 0
        and 1.

    Notes
    -----
    - The model should be a TensorFlow Keras Model instance.
    - The true labels `truth` and the predicted values `preds` should have the same
      shape.

    preds = np.array(tf.squeeze(preds))
    truth = np.array(truth)
    """
    
    # Reformat predictions from a softmax output
    if model.output_act == "softmax":
        bin_preds = [np.argmax(elem) for elem in preds]
        truth = np.array([np.argmax(elem) for elem in truth])

    # Reformat predictions from a sigmoid output
    if model.output_act == "sigmoid":
        bin_preds = [0 if elem < 0.5 else 1 for elem in preds]
    
    correct = (truth == bin_preds)
    accuracy = correct.sum() / correct.size

    return accuracy



def train_network(model, train_dataset,
                  train_epochs, optimizer, classification=True):

    """
    Train the specified model using the provided training dataset and optimizer.

    This function performs training on the specified model using the given training
    dataset and optimizer. It iterates for a specified number of training epochs,
    updating the model's weights based on the computed gradients. The function
    optionally supports classification tasks and reports accuracy metrics.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained. It should be a TensorFlow Keras Model object.
    train_dataset : tf.data.Dataset
        The training dataset containing input features and corresponding labels.
        It should be a TensorFlow Dataset object with tuples (X_train, y_train),
        where X_train represents the input data and y_train is the corresponding
        label or target.
    train_epochs : int
        The number of training epochs. The model will be trained for this many
        epochs over the entire training dataset.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use for updating the model weights during training.
        It should be a TensorFlow Keras Optimizer object.
    classification : bool, optional
        A flag indicating whether the task is a classification task. If True, the
        function computes and reports the accuracy of the model during training.
        If False, the accuracy is not computed and "NA" will be displayed. 
        Default is True.

    Returns
    -------
    tf.keras.Model
        The trained model after completing the training process.

    Notes
    -----
    - The model should be a TensorFlow Keras Model instance and the optimizer
      should be a TensorFlow Keras Optimizer instance.
    - The training dataset should be a TensorFlow Dataset object containing
      tuple pairs (X_train, y_train) for input features and corresponding labels.
    """

    # Iterate for a number of training epochs.
    for epoch in range(1, train_epochs+1):
        total_loss = total_accuracy = 0
        
        # Iterate over batches.
        for batch in train_dataset:
            X_train = batch[0]
            y_train = batch[1]
            
            # Pass the training data through the model and make predictions.
            # Compute the loss, including the loss from regularization.
            # Get the gradients for all trainable variables.
            with tf.GradientTape() as tape:
                train_preds = model(X_train)
                loss = get_loss(
                        model, y_train, train_preds, 
                        reg_penalty=True, group_penalty=False)
                grads = tape.gradient(loss, model.trainable_variables)
            del tape
            
            # Obtain the accuracy (for classification only).
            if classification:
                accuracy = get_accuracy(model, y_train, train_preds)
                total_accuracy += accuracy
            else:
                accuracy = "NA"
            
            # Update the weights of the variables in the nwtwork.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Add to the total loss.
            total_loss += loss

        # Format and print the progress of training.
        if (epoch == 1) or (epoch % 10 == 0) or (epoch == train_epochs):
            epoch_loss = total_loss.numpy() / len(train_dataset)
            epoch_accuracy = total_accuracy / len(train_dataset)
            print_string = (
                    f"Epoch: {epoch}\t" +
                    f"Loss: {epoch_loss:.4f}\t" + 
                    f"Accuracy: {epoch_accuracy:.4f}")
            print(print_string)

    return model
    







def check_network(model, dG_init, drop_cols):
    """
    Prune the neural network architecture based on connection weights.

    This function analyzes the connection weights of the model's neural network
    architecture and prunes it by removing modules and edges with zero connection
    weights. The function also calculates the sparsity of the network.

    Parameters
    ----------
    model : YourModel
        The neural network model to be pruned. It should be an instance of the
        YourModel class.

    dG_init : networkx.DiGraph
        The initial directed graph representation of the neural network. This graph
        should include all modules and edges of the original network.

    drop_cols : dict
        A dictionary that maps module names to lists of child module names that are
        to be removed from the network. This is used to track modules with zero
        connection weights to their children in the input layer.

    Returns
    -------
    YourModel
        The pruned neural network model after removing modules and edges with zero
        connection weights.

    networkx.DiGraph
        The pruned directed graph representation of the neural network after removing
        modules and edges with zero connection weights.

    dict
        The updated drop_cols dictionary containing modules with zero connection
        weights to be removed from the network.

    float
        The sparsity of the neural network architecture as a percentage, which
        represents the proportion of zero connection weights in the network.

    Notes
    -----
    - The model should be an instance of the YourModel class.
    - The initial directed graph `dG_init` should include all modules and edges of the
      original network.
    - The `drop_cols` dictionary is updated in place with modules to be removed.
    """

    dG_prune = dG_init.copy()
    
    all_connects = list()
    for var in model.trainable_variables:
        update = False
        layer_name = var.name.split("/")[0]
        mod = layer_name.split("_")[0]
        layer_type = layer_name.split("_")[1]
        

        if layer_type not in ["inp", "mod", "aux"]:
            continue
        if "bias" in var.name:
            continue
        
        layer = model.network_layers[layer_name]
        connections = tf.cast(tf.not_equal(var, 0.), tf.float32)
        layer.connections = connections
        nonzero = tf.math.count_nonzero(connections)
        
        # If the AUX layer contains only 0s, remove the module from the
        # ontology.

        if "aux" in var.name:
            all_connects.extend(connections.numpy().flatten())
            if nonzero == 0:
                try:
                    dG_prune.remove_node(mod) 
                    update = True
                except:
                    nx.NetworkXError

        # Check to see which inputs have only zero weights connecting to the 
        # input layer.
        if layer_type == "inp":

            input_set = sorted(list(model.term_direct_input_map[mod]))
            row = model.network_dims.loc[model.network_dims["Module"] == mod]
            children = list(row["Children"])
            for i, (id, child) in enumerate(zip(input_set, children[0])):
                all_connects.append(int(connections[id][i]))
                nonzero = tf.math.count_nonzero(connections[id])

                
                if nonzero == 0:
                    drop_cols[mod].append(child)
                    drop_cols[mod] = list(set(drop_cols[mod]))
                    try:
                        dG_prune.remove_edge(mod, child)
                        update = True
                    except:
                        nx.NetworkXError
        # Deal with the module layers
        if layer_type == "mod":

            all_connects.extend(connections.numpy().flatten())
            row = model.network_dims.loc[model.network_dims["Module"] == mod]
            children = list(row["Children"])
            
            start_index = 0
            for i, child in enumerate(children[0]):
                child_row = (
                        model.network_dims.loc[model.network_dims["Module"] == child])
                aux_enabled = child_row.iloc[0]["Aux_enabled"]
                child_neuron_num = child_row.iloc[0]["Neurons"]
                child_neuron_num = 1 if aux_enabled else child_neuron_num
                array = connections.numpy()
                child_array = array[start_index : start_index + child_neuron_num]
                nonzero = tf.math.count_nonzero(child_array)
                if nonzero == 0:
                    drop_cols[mod].append(child) 
                    drop_cols[mod] = list(set(drop_cols[mod]))
                    try:
                        dG_prune.remove_edge(mod, child)
                        update = True
                    except:
                        nx.NetworkXError
                start_index += child_neuron_num


    all_connects = [int(num) for num in all_connects]
    zeros = all_connects.count(0)
    sparsity = round((zeros / len(all_connects) * 100), 3)
    return model, dG_prune, drop_cols, sparsity
        







def prune_network(model, X, y, train_dataset, prune_epochs, 
                  optimizer, gl_pen1, l0_pen1, gl_pen2, l0_pen2):
    """
    Prune the neural network architecture using the sparse group lasso method.

    This function prunes the neural network model based on the sparse group lasso
    method. It iteratively applies the pruning algorithm to remove connections
    with low importance while preserving the model's performance. The pruning process
    is performed for a specified number of pruning epochs.

    Parameters
    ----------
    model : YourModel
        The neural network model to be pruned. It should be an instance of the
        YourModel class.

    X : ndarray
        The input data for training the model.

    y : ndarray
        The target labels corresponding to the input data `X`.

    train_dataset : tf.data.Dataset
        The training dataset used for optimization.

    prune_epochs : int
        The number of pruning epochs. Pruning will be applied iteratively for the
        specified number of epochs.

    optimizer : tf.keras.optimizers.Optimizer
        The optimizer used for training the neural network.

    gl_pen1 : float
        The penalty for the first group lasso term.

    l0_pen1 : float
        The penalty for the first L0 regularization term.

    gl_pen2 : float
        The penalty for the second group lasso term.

    l0_pen2 : float
        The penalty for the second L0 regularization term.

    Returns
    -------
    YourModel
        The pruned neural network model after applying the sparse group lasso pruning.

    Notes
    -----
    - The model should be an instance of the YourModel class.
    - The input data `X` and target labels `y` are used to perform pruning updates.
    - The training dataset `train_dataset` is used to obtain batches for optimization.
    - The specified optimizer `optimizer` is used to update the model during pruning.
    - The `gl_pen1`, `l0_pen1`, `gl_pen2`, and `l0_pen2` are hyperparameters that control
      the strength of different regularization terms in the pruning algorithm.
    """

    for prune_epoch in range(prune_epochs):
        for batch in train_dataset:
            X = batch[0]
            y = batch[1]

            with tf.GradientTape() as tape:
                tape.reset()
                train_preds = model(X)
                trainable_vars = model.trainable_variables
                loss = get_loss(
                        model, y, train_preds, 
                        reg_penalty=True, group_penalty=True)
                
                grads = tape.gradient(loss, trainable_vars)
            del tape
            



            #optimizer.apply_gradients(zip(grads, model.trainable_variables))
            old_parameters = [tf.stop_gradient(p) for p in model.trainable_variables]
            
            # Iterate over the trainable variables.
            for var, grad, old_param in zip(
                    model.trainable_variables, grads, old_parameters):
                var_name = var.name
                mod_name = var_name.split("_")[0].replace("-", ":")
                layer_name = var.name.split("/")[0]
                                
                fy = tf.constant(tf.stop_gradient(loss))
                





                with tf.GradientTape() as tape:
                    tape.reset()
                    trainable_vars = model.trainable_variables
                    loss = get_loss(model, model(X), y, reg_penalty=True)
                    grad = tape.gradient(loss, var)
                del tape
                


                L = 1
                for k in range(150):
                    
                    s = 1.0/(1.1*L)
                    u_new = sparse_group_lasso(
                            (old_param - s*tf.stop_gradient(grad)), 
                            s*gl_pen1,
                            s*l0_pen1,
                            s*gl_pen2,
                            s*l0_pen2)

                    var.assign(u_new)
                    var_diff = tf.math.subtract(tf.stop_gradient(u_new), old_param)
                    Q_L = (fy 
                            + tf.math.reduce_sum(tf.math.multiply(var_diff,
                                                                  grad)) 
                            + L/2 * tf.math.reduce_sum(tf.math.pow(var_diff, 2))
                          )
                    
                    with tf.GradientTape() as tape:
                        tape.reset()
                        loss = get_loss(model, model(X), y, reg_penalty=True)
                    del tape
                    
                    if loss <= Q_L:
                        break
                    L = 1.1*L

                

