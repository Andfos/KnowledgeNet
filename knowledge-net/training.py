import tensorflow as tf
import numpy as np
import networkx as nx
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import re







def sparse_group_lasso(w, lamb1, eta1, lamb2, eta2):
    """
    Apply Sparse Group Lasso regularization to a given weight matrix.

    This function updates the input weight matrix 'w' by inducing sparsity
    both within columns and within rows based on specified regularization
    parameters (lamb1, eta1, lamb2, eta2).
    
    Parameters
    ----------
    w : tf.Tensor
        The input weight matrix to be regularized.
    lamb1 : float
        The group lasso regularization strength for inducing sparsity of 
        columns.
    eta1 : float
        The l0 penalty applied to induce sparsity within columns.
    lamb2 : float
        The group lasso regularization strength for inducing sparsity of 
        rows.
    eta2 : float
        The l0 penalty applied to induce sparsity within rows.

    Returns
    -------
    tf.Tensor
        The updated weight matrix after applying Sparse Group Lasso regularization.

    Notes
    -----
    This function considers two types of regularization:
      1. Column sparsity: Each column is the set of all weights incoming to a 
         neuron. It encourages certain columns to contain
         all zeros (controlled by lamb1) or some zeros (controlled by 
         eta1).
      2. Row sparsity: Each row is the set of all weights outgoing from a
         neuron. It encourages certain rows to contain all
         zeros (controlled by lamb2) or some zeros (controlled by eta2).

    Examples
    --------
    >>> import tensorflow as tf
    >>> w = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
    >>> lamb1 = 0.1
    >>> eta1 = 0.01
    >>> lamb2 = 0.05
    >>> eta2 = 0.005
    >>> updated_w = sparse_group_lasso(w, lamb1, eta1, lamb2, eta2)

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
    Calculate the total loss for a given model's predictions.

    This function computes the total loss, which consists of the prediction loss
    and optionally includes a regularization penalty, if specified.

    Parameters
    ----------
    model : tf.keras.Model
        The neural network model for which to calculate the loss.
    y_true : tf.Tensor
        The true target values.
    y_pred : tf.Tensor
        The predicted values generated by the model.
    reg_penalty : bool, optional
        A flag indicating whether to include regularization penalties in the loss.
        If True (default), the regularization loss, if defined in the model's layers,
        will be added to the total loss. If False, only the prediction loss is
        considered.
        
    Returns
    -------
    loss : tf.Tensor
        The total loss, which can include both the prediction loss and regularization
        penalties, if enabled.
    """

    # Get prediction loss.
    y_pred = tf.cast(y_pred, tf.float32)
    loss = model.loss_fn(y_true=y_true, y_pred=y_pred)
    
    # Add in the regularization loss.
    if reg_penalty:
        loss += sum(model.losses)

    return loss




def get_accuracy(model, truth, preds):
    """
    Calculate the accuracy of model predictions.

    This function computes the accuracy of a machine learning model's predictions
    given the ground truth labels. It is designed to handle both softmax and
    sigmoid output activations.

    Parameters
    ----------
    model : tf.keras.Model 
        An instance of your machine learning model class.
    truth : tf.Tensor
        An tensor containing the ground truth labels.
    preds : tf.Tensor
        An tensor containing the model's predictions.

    Returns
    -------
    accuracy : float
        The accuracy of the model's predictions, represented as a decimal value
        between 0.0 and 1.0.
    """

    preds = np.array(tf.squeeze(preds))
    truth = np.array(truth)
    
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






def train_network(
        model, train_dataset,
        train_epochs, optimizer, classification=True):
    """
    Train a machine learning model using a given dataset and optimizer.

    This function performs model training for a specified number of epochs using
    the provided training dataset and optimizer. It includes forward and backward
    passes, gradient computation, weight updates, and optional accuracy monitoring
    for classification tasks.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained.
    train_dataset : tf.data.Dataset
        The training dataset, typically created using TensorFlow data pipeline
        utilities.
    train_epochs : int
        The number of training epochs (complete passes through the training dataset).
    optimizer : tf.keras.optimizers.Optimizer
        The optimization algorithm used for weight updates during training.
    classification : bool, optional
        A flag indicating whether the task is classification (True) or not (False).
        If True (default), the function monitors and prints accuracy during training.

    Returns
    -------
    model : tf.keras.Model
        The trained machine learning model.
    """
    
    # Iterate for a number of training epochs.
    for epoch in range(train_epochs):
        
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
                rounded_accuracy = str(round(accuracy, 3))
            else:
                rounded_accuracy = "NA"
            
            # Update the weights of the variables in the nwtwork.
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            # Format and print the progress of training.
            print_string = (
                    f"Step: {optimizer.iterations.numpy()}\t" +
                    f"Loss: {str(round(loss.numpy(), 3))}\t" + 
                    f"Accuracy: {rounded_accuracy}")
            print(print_string)

    return model
    







def check_network(model, dG_init, drop_cols):

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
    sparsity = round((zeros / len(all_connects) * 100), 2)
    return model, dG_prune, drop_cols, sparsity
        







def prune_network(
        model, X, y, train_dataset, prune_epochs, 
        optimizer, gl_pen1, l0_pen1, gl_pen2, l0_pen2):
    """
    Perform network pruning on KnowledgeNet model instance.

    This function conducts network pruning using a specified optimization technique
    and criteria over a set number of epochs. It updates the model's weights
    by enforcing sparsity in a structured manner.
    
    Parameters
    ----------
    model : tf.keras.Model
        An instance of your machine learning model class.
    X : tf.Tensor
        Input data used for pruning, typically a batch from the training dataset.
    y : tf.Tensor
        Ground truth labels corresponding to the input data.
    train_dataset : tf.data.Dataset
        The training dataset, used for iterating over batches during pruning.
    prune_epochs : int
        The number of pruning epochs, controlling the duration of the pruning process.
    optimizer : tf.keras.optimizers.Optimizer
        The optimization algorithm used for weight updates during pruning.
    gl_pen1 : float
        The regularization strength for Group Lasso of columns.
    l0_pen1 : float
        The L0 penalty applied to induce sparsity within columns.
    gl_pen2 : float
        The regularization strength for Group Lasso of rows.
    l0_pen2 : float
        The L0 penalty applied to induce sparsity within rows.

    Returns
    -------
    tf.keras.Model
        The machine learning model after network pruning.
    """
    
    # Prune the model for a specified number of epochs.
    for prune_epoch in range(prune_epochs):
        for batch in train_dataset:
            X = batch[0]
            y = batch[1]
            
            # Calculate gradients. 
            with tf.GradientTape() as tape:
                tape.reset()
                train_preds = model(X)
                trainable_vars = model.trainable_variables
                loss = get_loss(
                        model, y, train_preds, reg_penalty=True)
                grads = tape.gradient(loss, trainable_vars)
            del tape
            
            # Iterate over the trainable variables.
            old_parameters = [tf.stop_gradient(p) for p in model.trainable_variables]
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


