import tensorflow as tf
import numpy as np
#from penalties import *
import networkx as nx
from utils import debugger
import torch
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



def l0_glasso(w, lamb, eta):
    w = w.numpy()
    S = tf.zeros_like(w).numpy()
    
    # For updating biases.
    if len(w.shape)<2:
        S = tf.where(tf.math.abs(w) > lamb, 0.5 * (w - lamb)**2 * eta, S)
        w_new = tf.where(S > 0, (1- lamb/tf.math.abs(w)) * w, tf.zeros_like(w))
    
    # For updating weights.
    else:

        # Return a sorted kernel such that the first row (input) contains the 
        # maximum of the column (neuron). The index records the input with the 
        # maximum value of that column.
        sorted_w = tf.sort(tf.math.abs(w), axis=0, direction='DESCENDING')
        indices = tf.argsort(tf.math.abs(w), axis=0, direction='DESCENDING')

        for i in range(S.shape[0]):
            y_i = tf.norm(sorted_w[:i+1,:], ord="euclidean", axis=0)
            S[i,:] = tf.where(y_i>lamb, 0.5*(y_i-lamb)**2-(i+1)*eta, S[i,:])
        
        values = tf.math.reduce_max(S, axis=0)
        ind = tf.math.argmax(S, 0)
        
        w_new = tf.zeros_like(w).numpy() 
        for i in range(w.shape[1]):
            if values[i]>0:
                w_new[indices[:ind[i]+1,i], i] = w[indices[:ind[i]+1,i], i]

        col_norm = tf.norm(w_new, ord="euclidean", axis=0)
        norm_coef = tf.where(col_norm == 0, tf.zeros_like(col_norm), 1-lamb/col_norm)
        w_new = w_new * norm_coef
    
    return w_new


def l0_glasso_row(w, lamb, eta):
    w = w.numpy()
    S = tf.zeros_like(w).numpy()
    
    # For updating biases.
    if len(w.shape)<2:
        S = tf.where(tf.math.abs(w) > lamb, 0.5 * (w - lamb)**2 * eta, S)
        w_new = tf.where(S > 0, (1- lamb/tf.math.abs(w)) * w, tf.zeros_like(w))
    
    # For updating weights.
    else:

        # Return a sorted kernel such that the first row (input) contains the 
        # maximum of the column (neuron). The index records the input with the 
        # maximum value of that column.
        """
        sorted_w = tf.sort(tf.math.abs(w), axis=1, direction='DESCENDING')
        indices = tf.argsort(tf.math.abs(w), axis=1, direction='DESCENDING')

        for i in range(S.shape[1]):
            y_i = tf.norm(sorted_w[:, :i+1], ord="euclidean", axis=1)
            S[:,i] = tf.where(y_i>lamb, 0.5*(y_i-lamb)**2-(i+1)*eta, S[:,i])
        
        values = tf.math.reduce_max(S, axis=1)
        ind = tf.math.argmax(S, 1)
        
        w_new = tf.zeros_like(w).numpy() 
        for i in range(w.shape[0]):
            if values[i]>0:
                w_new[i, indices[i, :ind[i]+1]] = w[i, indices[i, :ind[i]+1]]
        """
        


        #S = tf.zeros_like(w).numpy()
        sorted_w = tf.sort(tf.math.abs(w), axis=1, direction='DESCENDING')
        indices = tf.argsort(tf.math.abs(w), axis=1, direction='DESCENDING')
        for i in range(S.shape[1]):
            y_i = tf.norm(sorted_w[:i+1, ], ord="euclidean", axis=0)
            S[:, i] = tf.where(y_i>lamb, 0.5*(y_i-lamb)**2-(i+1)*eta, S[:, i])

        values = tf.math.reduce_max(S, axis=0)
        ind = tf.math.argmax(S, 0)

        w_new = tf.zeros_like(w).numpy() 

        for i in range(w.shape[0]):
            if values[i]>0:
                w_new[indices[:ind[i] + 1, i], i] = w[indices[:ind[i] + 1, i], i]
        



        col_norm = tf.norm(w_new, ord="euclidean", axis=0)
        norm_coef = tf.where(col_norm == 0, tf.zeros_like(col_norm), 1-lamb/col_norm)
        w_new = w_new * norm_coef






    return w_new


def l0_glasso_fast(w, lamb, eta):
    w = w.numpy()
    S = tf.zeros_like(w).numpy()
    
    # For updating biases.
    if len(w.shape)<2:
        S = tf.where(tf.math.abs(w) > lamb, 0.5 * (w - lamb)**2 * eta, S)
        w_new = tf.where(S > 0, (1- lamb/tf.math.abs(w)) * w, tf.zeros_like(w))
    
    # For updating weights.
    else:
        w_ones = tf.ones_like(w).numpy()
        w_zeros = tf.zeros_like(w).numpy()
        c_diag = tf.cast(tf.linalg.diag(tf.range(0, w.shape[1])), tf.float32)
        cols = tf.matmul(w_ones,c_diag)
        sorted_w = tf.sort(tf.math.abs(w), axis=0, direction='DESCENDING')
        indices = tf.argsort(tf.math.abs(w), axis=0, direction='DESCENDING')
        lower_tri = tf.linalg.band_part(
                tf.ones((w.shape[0], w.shape[0])), -1, 0)
        #y_i = torch.sqrt(torch.mm(lower_tri,sorted_u**2)) 
        y_i = tf.math.sqrt(tf.matmul(lower_tri, sorted_w**2))
        #S = torch.where(y_i>lamb, 0.5*(y_i - lamb)**2 - eta*torch.mm(torch.diag(torch.arange(1,u.shape[0]+1)).to(device).float(),u_ones), S)
        tri = tf.cast(tf.linalg.diag(tf.range(1, w.shape[0]+1)),tf.float32)
        S = tf.where(y_i > lamb,
                     0.5*(y_i - lamb)**2 - eta*tf.matmul(tri, w_ones),
                     S)
        values = tf.math.reduce_max(S, axis=0)
        ind = tf.math.argmax(S, 0)
        indx = tf.where(values>0, ind+1, tf.zeros_like(ind))
        #ind0 = torch.where(torch.mm(u_ones,torch.diag(indx).to(device).float())<torch.mm(torch.diag(torch.arange(1,u.shape[0]+1)).to(device).float(),u_ones),u_zeros,u_ones)
        ind0 = tf.where(tf.matmul(w_ones, tf.cast(tf.linalg.diag(indx),
                                                  tf.float32))
                        < tf.matmul(tri, w_ones),
                        w_zeros, w_ones)
        w_zeros[indices, tf.cast(cols, tf.int64)] = sorted_w * ind0
        #u_new_unnormalized = u_zeros*torch.sign(u)
        w_new_unnormalized = w_zeros*tf.math.sign(w)

        ## Normalization by column (regular)
        col_norm = tf.norm(w_new_unnormalized, ord="euclidean", axis=0)
        norm_coef_1 = tf.where(col_norm == 0, tf.zeros_like(col_norm), 1 - lamb/col_norm)
        w_new = w_new_unnormalized * norm_coef_1
        
        # Normalization by row (experimental)
        #w_new = w_new.numpy()
        #col_norm = tf.norm(w_new, ord="euclidean", axis=1)
        #norm_coef_2 = tf.where(col_norm == 0, tf.zeros_like(col_norm), 1 - lamb/col_norm)
        #w_new = tf.zeros_like(w_new_unnormalized).numpy()
        #for index in range(len(norm_coef_2)):
            #w_new[index] = w_new_unnormalized[index] * norm_coef_2[index]
            #w_new[index] = w_new[index] * norm_coef_2[index]
    
    return w_new







def get_loss(model, y_true, y_pred, reg_penalty=True, group_penalty=False):
    
    if model.output_act == "softmax":
        pass
        #y_pred = tf.math.argmax(y_pred, 1)
        #y_pred = tf.expand_dims(y_pred, axis=1)
    y_pred = tf.cast(y_pred, tf.float32)
    #y_true = tf.expand_dims(y_true, axis=1)

    loss = model.loss_fn(y_true=y_true, y_pred=y_pred)
    # Add in the regularization loss.
    if reg_penalty:
        loss += sum(model.losses)

    # Add in the group lasso loss term.
    if group_penalty:
        for i, var in enumerate(model.trainable_variables):
            if var.shape[0] > 1 and "inp" not in var.name:
                gl_loss = tf.math.multiply(
                        tf.math.reduce_sum(
                                tf.norm(var, ord = "euclidean", axis=1)), 
                                0.5)
                loss = tf.math.add(loss, gl_loss)
    return loss




def get_accuracy(model, truth, preds):
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









    





















def train_network(model, train_dataset,
                  train_epochs, optimizer, classification=True):
    
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

            """
            for child_mask, child in zip(connections, children[0]):
                print(var.name)
                print(child_mask)
                print(children)
                raise

                nonzero = tf.math.count_nonzero(child_mask)
                if nonzero == 0:
                    drop_cols[mod].append(child) 
                    drop_cols[mod] = list(set(drop_cols[mod]))
                    try:
                        dG_prune.remove_edge(mod, child)
                        update = True
                    except:
                        nx.NetworkXError"""

    all_connects = [int(num) for num in all_connects]
    zeros = all_connects.count(0)
    sparsity = round((zeros / len(all_connects) * 100), 2)
    return model, dG_prune, drop_cols, sparsity
        







def prune_network(model, X, y, train_dataset, prune_epochs, 
                  optimizer, gl_pen, l0_pen):

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
                        reg_penalty=True, group_penalty=False)
                
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
                    u_new = l0_glasso_fast((old_param - s*tf.stop_gradient(grad)), 
                                           s*gl_pen,
                                           s*l0_pen)
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
                    
                    #print(k)
                    #print(var.name)
                    if loss <= Q_L:
                        #print(f"loss: {loss}")
                        break
                    L = 1.1*L

                
            #print(model.get_weights())        

