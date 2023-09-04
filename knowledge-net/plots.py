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



def construct_ontology(dG, root, title="Ontology", module_file=None):
    
    onto_df = pd.DataFrame()

    source_ids = list()
    source_names = list()
    source_types = list()
    target_ids = list()
    target_names = list()
    target_types = list()
    edge_values = list()
    titles = list()

    # Collect edge information from the ontology. 
    for parent, child in dG.edges:
        source_ids.append(child)
        source_names.append(child)
        source_types.append(child)
        target_id = ("Root" if parent == root else parent)
        target_ids.append(target_id)
        target_names.append(parent)
        target_types.append(parent)
        edge_values.append(1)
        titles.append("NA")
    
    titles[0] = title
    # Load the lists into a dataframe.
    onto_df["source_id"] = source_ids
    onto_df["source_name"] = source_names
    onto_df["source_type"] = source_types
    onto_df["target_id"] = target_ids
    onto_df["target_name"] = target_names
    onto_df["target_type"] = target_types
    onto_df["edge_value"] = edge_values
    onto_df["title"] = titles
    


    if module_file is not None:
        module_df = pd.read_csv(module_file)
        modules = list(module_df.iloc[:, 0])
        inputs = list(module_df.iloc[:, 1])
        inputs = [inp.lower() for inp in inputs]
        for mod, inp in zip(modules, inputs):
            ind = onto_df.index[(onto_df["source_name"] == inp)]
            if len(ind) != 0:
                ind = ind[0]
                onto_df.at[ind, "source_type"] = mod 



    return onto_df








def construct_network(model, sub_dG, classes, title="Network", module_file=None):
    
    #cluster = "min_crossover"
    cluster = None

    # Initialize a dataframe and lists to record the architecture of the
    # network.
    network_df = pd.DataFrame()
    source_ids = list()
    source_names = list()
    source_types = list()
    target_ids = list()
    target_names = list()
    target_types = list()
    edge_values = list()
    y_coords = dict()
    
    # Iterate through the models trainable variables.
    for var in model.trainable_variables:
        array = tf.cast(var, tf.float32).numpy()
        layer_name = var.name.split("/")[0]
        mod = layer_name.split("_")[0]
        layer_type = layer_name.split("_")[1]
        
        # If the module is not in the pruned ontology, continue onto the next
        # module.
        if mod not in sub_dG.nodes:
            continue

        # Only record the architecture for input layers, module layers, and 
        # auxiliary layers.
        if layer_type not in ["inp", "mod", "aux"]:
            continue
        if "bias" in var.name:
            continue

        # Ensure the zeroed weights are exactly 0.
        layer = model.network_layers[layer_name]
        array = array * layer.connections.numpy()
        connections = layer.connections
        
        # Record the architecture of an auxiliary layer.
        if "aux" in layer_type:
            for i, w in enumerate(array):
                source_ids.append(f"{mod}_{i}")
                source_names.append(mod)
                source_types.append(mod)
                target_ids.append(layer_name)
                target_names.append(layer_name)
                target_types.append(layer_type)
                edge_values.append(w[0])
                
        # Record the architecture of an input layer.
        elif "inp" in layer_type:

            input_set = sorted(list(model.term_direct_input_map[mod]))
            row = model.network_dims.loc[model.network_dims["Module"] == mod]
            children = list(row["Children"])
            for i, (id, child) in enumerate(zip(input_set, children[0])):
                if child not in list(sub_dG.neighbors(mod)):
                    continue

                source_ids.append(f"{child}_input")
                source_names.append(child)
                source_types.append("Input")
                target_ids.append(f"{child}_0")
                target_names.append(child)
                target_types.append(child)
                w = array[id][i]
                edge_values.append(w)


        # If the current layer is a module layer...
        elif "mod" in layer_type:
            row = model.network_dims.loc[model.network_dims["Module"] == mod]
            children = list(row["Children"])
            

            # Peform TSNE on the weights.
            #weights_abs = pd.DataFrame(np.transpose(abs(array)))
            #weights_abs.columns = children
            #weights_abs = pd.DataFrame(abs(array))
            
            #tsne = TSNE(n_components=1, perplexity=1, n_iter=300)
            #tsne_results = tsne.fit_transform(connections)
            #normed = 20 * ((tsne_results - tsne_results.mean()) / tsne_results.std())

            start_index = 0
            node_counter = 0
            for child in children[0]:
            


                child_row = (
                        model.network_dims.loc[model.network_dims["Module"] == child])
                child_neuron_num = child_row.iloc[0]["Neurons"]
                aux_enabled = child_row.iloc[0]["Aux_enabled"]
                child_neuron_num = (1 if aux_enabled else child_neuron_num)
                if child not in list(sub_dG.neighbors(mod)):
                    start_index += child_neuron_num
                    continue
                
                child_array = array[start_index : start_index + child_neuron_num]
                for inp_neuron in range(child_neuron_num):
                    
                    for out_neuron in range(len(child_array[inp_neuron])):
                        if aux_enabled:
                            source_id = f"{child}_aux"
                            source_name = f"{child}_aux"
                            source_type = "aux"
                        else:
                            source_id = f"{child}_{inp_neuron}"
                            source_name = child
                            source_type = child
                        
                        #if mod in model.term_direct_input_map.keys():
                        #    y_coords[source_id] = normed[node_counter]
                        #    y_coords[f"{child}_input"] = normed[node_counter]


                        if mod == model.root:
                            target_id = f"Root_{out_neuron}"
                            target_name = f"{classes[out_neuron]}"
                        else:
                            target_id = f"{mod}_{out_neuron}"
                            target_name = mod

                        source_ids.append(source_id)
                        source_names.append(source_name)
                        source_types.append(source_type)
                        target_ids.append(target_id)
                        target_names.append(target_name)
                        target_types.append(mod)
                        w = child_array[inp_neuron][out_neuron]
                        edge_values.append(w)
                    node_counter += 1
                start_index += child_neuron_num        

    placeholder_coors = ["NA"] * len(source_ids) 
    network_df["source_id"] = source_ids
    network_df["source_name"] = source_names
    network_df["source_type"] = source_types
    network_df["target_id"] = target_ids
    network_df["target_name"] = target_names
    network_df["target_type"] = target_types
    edge_values = [abs(w) for w in edge_values]
    network_df["edge_value"] = edge_values
    titles = ["NA"] * len(source_ids)
    network_df["title"] = titles
    placeholder_coords = ["NA"] * len(source_ids)
    network_df["y_coord"] = placeholder_coords
    


    

    for source, y_coord in y_coords.items():
        indices = network_df.index[(network_df["source_id"] == source)]
        for ind in indices:
            network_df.at[ind, "y_coord"] = y_coord[0]

    # Drop connections with edge weight of 0.
    network_df = network_df[~(network_df["edge_value"] == 0)]
    
    df_init = network_df.copy()
    while True:
        source_ids_set = set(list(network_df["source_id"]))
        target_ids_set = set(list(network_df["target_id"]))
        drop_source = source_ids_set.difference(target_ids_set)
        drop_source = [s for s in drop_source if not "input" in s]
        drop_target = target_ids_set.difference(source_ids_set)
        drop_target = [t for t in drop_target if not "Root" in t]
        
        network_df = network_df[~network_df["source_id"].isin(drop_source)]
        network_df = network_df[~network_df["target_id"].isin(drop_target)]

        if len(network_df.index) == len(df_init.index):
            break
        df_init = network_df.copy()
    





    #pd.set_option('display.max_rows', None)

    
    
    if cluster == "tsne":
        target_ids = list(network_df["target_id"])
        target_ids_set = sorted(set(target_ids), key = target_ids.index)
        for target in target_ids_set:
            if "Root" in target:
                continue
            temp = network_df[network_df["target_id"] == target]
            w_av = sum(temp["y_coord"] * temp["edge_value"]) / sum(temp["edge_value"])
            network_df.loc[network_df["source_id"] == target, "y_coord"] = w_av

    
    # Cluster the modules to minimze the number of crossing lines.
    if cluster == "min_crossover":
        rev_df = network_df.iloc[::-1]
        root_nodes = rev_df[rev_df["target_id"].str.contains("Root")]
        root_ids = list(root_nodes["target_id"])
        root_nums = set([int(target.split("_")[1]) for target in root_ids])
        mid_root = stats.median(root_nums)

        
        target_ids = list(rev_df["target_id"])
        target_ids_set = sorted(set(target_ids), key = target_ids.index)
        for target in target_ids_set:
            temp = network_df[network_df["target_id"] == target]
            if "Root" in target:
                node_num = int(target.split("_")[1])
                y_coord = (node_num - mid_root) * len(temp) * 5
                #y_coord = node_num * 100
                rev_df.loc[rev_df["target_id"] == target, "y_coord"] = y_coord
            else:
                temp2 = rev_df.loc[rev_df["source_id"] == target]

                w_av = (
                        sum(temp2["y_coord"] * temp2["edge_value"]) / 
                        sum(temp2["edge_value"]))
                
                rev_df.loc[rev_df["target_id"] == target, "y_coord"] = w_av
                network_df.loc[network_df["source_id"] == target, "y_coord"] = w_av

    
        input_nodes = network_df[network_df["source_id"].str.contains("input")]
        input_nodes = input_nodes.sort_values(by=["y_coord"], axis=0)
        mid_rank = (len(input_nodes) - 1) / 2
        
        
        rank = 0
        for ind, row in input_nodes.iterrows():
            current_coord = row["y_coord"]
            new_coord = current_coord + rank - mid_rank
            rank += 1
            network_df.at[ind, "y_coord"] = new_coord



    if module_file is not None:
        module_df = pd.read_csv(module_file)
        modules = list(module_df.iloc[:, 0])
        inputs = list(module_df.iloc[:, 1])
        inputs = [inp.lower() for inp in inputs]
        for mod, inp in zip(modules, inputs):
            indices = network_df.index[(network_df["source_name"] == inp)]
            for ind in indices:
                network_df.at[ind, "source_type"] = mod 
            indices = network_df.index[(network_df["target_name"] == inp)]
            for ind in indices:
                network_df.at[ind, "target_type"] = mod 

    # Set the title of the plot in the top row of the "title" column.
    network_df.at[0, "title"] = title
    return network_df
        





# From hitvoice
# (https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7)
def cm_analysis(y_true, y_pred, filename, labels, 
                title="Placeholder for title", ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    
    Parameters
    ----------
    y_true : True label of the data, with shape (nsamples,).
    y_pred : Prediction of the data, with shape (nsamples,).
    filename :  Filename of figure file to save.
    labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
    ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
    figsize:   the size of the figure plotted.
    """

    y_pred = [np.argmax(elem) for elem in y_pred]
    y_true = [np.argmax(elem) for elem in y_true]
    y_pred = [labels[ind] for ind in y_pred]
    y_true = [labels[ind] for ind in y_true]
    
    if ymap is not None:



        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    #cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    
    cm_mat = cm.values
    off_diag_mask = np.eye(*cm_mat.shape, dtype=bool)
    #print(off_diag_mask)


    #sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="PiYG")

    sns.heatmap(cm, annot=annot, mask=~off_diag_mask, 
                cmap='RdBu', fmt="", ax=ax, 
                vmin=100/len(labels), vmax=100, cbar_kws={"label":"Correct"})
    sns.heatmap(cm, annot=annot, mask=off_diag_mask, 
                cmap='Reds', fmt="", ax=ax, 
                vmin=0, vmax=100, cbar_kws={"label":"Incorrect"})
    #sns.heatmap(cm, annot=True, cmap='OrRd')
    plt.suptitle("Classification Matrix", fontsize=24)
    plt.title(title, fontsize=16)

    plt.savefig(filename)
    plt.close()




