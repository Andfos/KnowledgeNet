import shutil
from config import *
import pandas as pd
import keras
import sys 
from tensorflow.keras import initializers
import datetime
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
from utils import *
from networks import KnowledgeNet
from training import (
        get_loss, train_network, prune_network, check_network, get_accuracy)
from plots import construct_ontology, construct_network, cm_analysis
import networkx as nx
import math 
from tensorflow.keras import regularizers 
from sklearn import preprocessing
import pickle
import shap







def run_KnowledgeNet():
    # Save the relevant data in the correct location
    #shutil.copyfile("config.py", f"{RES_DIR}/config.py")

    # If running the function in BUILD_MODE, generate data according to user 
    # specified function.
    if BUILD_MODE:
        function_name = f"f(X) = {FUNC} + E~N(0, {NOISE_SD})"
        X, y = generate_data(
                function=FUNC,
                noise_sd_func=NOISE_SD,
                data_size=DATA_SIZE,
                input_dim=INPUT_DIM,  
                lower=LOWER,
                upper=UPPER)
        classes = ["Output"]


    # If running in PRODUCTION_MODE (aka not BUILD_MODE), load the data from file.
    else:
        data = pd.read_csv(DATA_FILE, sep="\t")
        X = data.iloc[:, 0:-1].to_numpy()
        y = data.iloc[:, -1:].to_numpy()
        le = preprocessing.LabelEncoder()
        
        #y = y.ravel()
        le.fit(y)
        y = le.transform(y)
        classes = le.classes_

        if OUTPUT_ACT == "softmax":
            y = np_utils.to_categorical(y)


    # Split the training and test data. Batch the dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=TEST_SIZE,
                                                        random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(BATCH_SIZE)


    # Load the ontology.
    input_id_map = load_mapping(f"{EXP_DIR}/data/features.tsv")
    dG, root, term_size_map, term_direct_input_map = load_ontology(
            f"{EXP_DIR}/data/ontology.tsv", 
            input_id_map)


    # Set the optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam')


    # Load a pretrained model or initialize a new one.
    model = KnowledgeNet(
            output_dim = OUTPUT_DIM,
            output_act = OUTPUT_ACT,
            module_act = MODULE_ACT,
            input_act = INPUT_ACT,
            root=root, 
            dG=dG,
            module_neurons_func=MODULE_NEURONS_FUNC, 
            input_dim=INPUT_DIM, 
            term_direct_input_map=term_direct_input_map,
            mod_size_map=term_size_map, 
            initializer=WEIGHTS_INIT, 
            input_regularizer=INPUT_REG,
            module_regularizer=MODULE_REG,
            loss_fn=LOSS_FN,
            aux=AUX,
            batchnorm=BATCHNORM) 


    if LOAD_MODEL:
        model.load_weights(f"{MODEL_SAVEDIR}/model")
        
    # Compile and build the model.
    model.compile(optimizer=optimizer, loss=LOSS_FN)
    model.build(input_shape = (BATCH_SIZE, INPUT_DIM))
    model.summary()


    # Fit the model if not trained yet. Save it if it converges to solution.
    if not LOAD_MODEL:
        model = train_network(
                model, 
                train_dataset, 
                train_epochs=TRAIN_EPOCHS, 
                optimizer=optimizer,
                classification=CLASSIFICATION)


    # Check the network structure.
    drop_cols = dict([(mod, []) for mod in model.mod_size_map.keys()])
    model, dG_prune, drop_cols, sparsity = check_network(
            model, model.dG, drop_cols)

    # Get loss on the train and test data prior to pruning.
    train_preds = model(X_train, batch_train=False)
    test_preds = model(X_test, batch_train=False)

    train_loss = round(
            get_loss(model, y_train, train_preds, reg_penalty=False).numpy(), 
            2)
    test_loss = round(
            get_loss(model, y_test, test_preds, reg_penalty=False).numpy(), 
            2)

    # Report accuracies if the problem is a classification problem.
    if CLASSIFICATION:
        train_acc = get_accuracy(model, y_train, train_preds)
        test_acc = get_accuracy(model, y_test, test_preds)
        train_metric_name = "TrainAcc"
        test_metric_name = "TestAcc"
        train_metric = round(train_acc*100, 3)
        test_metric = round(test_acc*100, 3)

    else:
        train_acc = "NA"
        test_acc = "NA"
        train_metric_name = "TrainLoss"
        test_metric_name = "TestLoss"
        train_metric = round(train_loss, 3)
        test_metric = round(test_loss, 3)

    # Create a base savefile for all result reporting.
    base_savefile = (
            "Preprune" +  
            f".{train_metric_name}_{str(train_metric)}" + 
            f".{test_metric_name}_{str(test_metric)}")

    # Write the ontology and network-graph to csv.
    base_title = (
            "Unpruned\t\t\t" + 
            f"{train_metric_name} : {str(train_metric)}\t\t\t" +                           
            f"{test_metric_name} : {str(test_metric)}")


    #onto_df = construct_ontology(
    #        dG, model.root, title=f"Ontology\t\t{base_title}",
    #        module_file=MODULE_FILE)
    #onto_df.to_csv(f"{RES_DIR}/Ontology_{base_savefile}.csv")

    #network_df = construct_network(
    #        model, model.dG, classes, title=f"Network:\t{base_title}", 
    #        module_file=MODULE_FILE)
    #network_df.to_csv(f"{RES_DIR}/Network_{base_savefile}.csv")


    if CLASSIFICATION:
        cm_analysis(
                y_test, test_preds, f"{RES_DIR}/Class_{base_savefile}.png", 
                labels=classes, title=base_title)

    # Save the trained model.
    if SAVE_MODEL:
        model.save_weights(f"{MODEL_SAVEDIR}/model")
        print(f"Saved model weights to {MODEL_SAVEDIR}/model")




    # Initialize a scores dataframe.
    scores_df = pd.DataFrame()
    init_sources = list()
    init_targets = list()
    init_scores = list()

    for target, source in model.dG.edges:
        init_sources.append(source)
        init_targets.append(target)
        init_scores.append(len(model.dG.edges))
    scores_df["Source"] = init_sources
    scores_df["Target"] = init_targets
    scores_df["Score"] = init_scores




    # Prune the unneeded modules.
    dG_current = model.dG
    score = 1
    pruning_period = 1
    init_sparsity = 0.0
    sparsity = 0.0
    for prune_train_iter in range(0, PRUNE_TRAIN_ITERATIONS):
        update = False
        retrain = False
        # Set the penalties for pruning columns.
        if pruning_period == 1:
            GL_PEN1 = 0.0
            L0_PEN2 = 0.0

            # Increment the group lasso penalty for columns.
            if GL_PEN2 < MAX_GL2:
                GL_PEN2 *= 1.05
            else:
                GL_PEN2 = MAX_GL2
            
            # Increment the L0 penalty for columns.
            if L0_PEN1 < MAX_L01:
                L0_PEN1 *= 1.1
            else:
                L0_PEN1 = MAX_L01

        # Set the penalties for pruning rows.
        if pruning_period == 2:
            L0_PEN2 = 0.0
            GL_PEN2 = 0.0
            # Increment the group lasso penalty for rows.
            if GL_PEN1 < MAX_GL1:
                GL_PEN1 *= 1.05
            else:
                GL_PEN1 = MAX_GL1
            
            # Increment the L0 penalty for rows.
            if L0_PEN1 < MAX_L01:
                L0_PEN1 *= 1.1
            else:
                L0_PEN1 = MAX_L01

            
        
        # Prune the network weights. 
        prune_network(model, X_train, y_train,
                      train_dataset, optimizer=optimizer,
                      gl_pen1=GL_PEN1, l0_pen1=L0_PEN1, 
                      gl_pen2=GL_PEN2, l0_pen2=L0_PEN2,
                      prune_epochs=1)
        
        
        # Check the network structure.
        model, dG_prune, drop_cols, sparsity = check_network(
                model, dG_current, drop_cols)
        

        # Update the user on the progress of pruning every UPDATE_ITERS.
        if (prune_train_iter % UPDATE_ITERS) == 0 and (prune_train_iter != 0):
            update = True
            if init_sparsity != sparsity:
                retrain = True
                init_sparsity = sparsity
            else:
                if GL_PEN2 == MAX_GL2 and L0_PEN1 == MAX_L01:
                    pruning_period = 2
                    MAX_GL1 *= 1.5
                    MAX_L01 *= 1.5
                    MAX_GL2 *= 1.5
                    MAX_L02 *= 1.5
                    MAX_L01 = 0.999
                if GL_PEN1 == 0.0 or L0_PEN1 == 0.0:
                    GL_PEN1 = 0.05
                    L0_PEN1 = 0.05
                if GL_PEN1 == MAX_GL1 and L0_PEN1 == MAX_L01:
                    MAX_GL1 *= 1.5
                    MAX_L01 *= 1.5
                    MAX_GL2 *= 1.5
                    MAX_L02 *= 1.5


        # Update the graphs and retrain (if RETRAIN) if the ontology has changed.
        if (dG_current.number_of_nodes() != dG_prune.number_of_nodes() 
            or dG_current.number_of_edges() != dG_prune.number_of_edges()):
                update = True
                retrain = True

        # Retrain the model (if UPDATE is True and RETRAIN is enabled).
        if retrain:
            model = train_network(
                    model, 
                    train_dataset, 
                    train_epochs=RETRAIN_EPOCHS, 
                    optimizer=optimizer,
                    classification=CLASSIFICATION)


        
        # Get loss on training and test data.
        train_preds = model(X_train, batch_train=False)
        test_preds = model(X_test, batch_train=False)
        train_loss = round(
                get_loss(model, y_train, model(X_train), reg_penalty=False).numpy(), 
                2)
        test_loss = round(
                get_loss(model, y_test, model(X_test), reg_penalty=False).numpy(), 
                2)

        # Obtain accuracy (if applicable) and set the metrics to be reported in 
        # titles and file names.
        if CLASSIFICATION:
            train_acc = get_accuracy(model, y_train, train_preds)
            test_acc = get_accuracy(model, y_test, test_preds)
            train_metric = round(train_acc*100, 3)
            test_metric = round(test_acc*100, 3)

        else:
            train_metric = round(train_loss, 3)
            test_metric = round(test_loss, 3)

        # Print the progress of pruning.
        print(f"Pruning iteration: {prune_train_iter}")
        print(f"Training loss: {str(train_loss)}\tTest loss: {str(test_loss)}")
        print(f"Train accuacy: {train_acc}\tTest Acc: {test_acc}")
        print(f"Sparsity: {str(sparsity)}%")
        print(f"Group lasso penalty 1: {round(GL_PEN1, 4)}\tL0 penalty 1: {round(L0_PEN1, 4)}")
        print(f"Group lasso penalty 2: {round(GL_PEN2, 4)}\tL0 penalty 2: {round(L0_PEN2, 4)}")
        print(f"Init sparsity {str(init_sparsity)}%")
        print("\n\n")
        

        # Save the ontology and network structure to csv files for viewing later.
        if update:
            
            
            if SAVE_MODEL:
                model.save_weights(f"{MODEL_SAVEDIR}/model")
                print(f"Saved model weights to {MODEL_SAVEDIR}/model")

            base_savefile = (
                    f".Sparsity_{str(sparsity)}" + 
                    f".{train_metric_name}_{str(train_metric)}" + 
                    f".{test_metric_name}_{str(test_metric)}")
            
            base_title = (
                    f"Sparsity : {str(sparsity)}    " + 
                    f"{train_metric_name} : {str(train_metric)}    " +                           
                    f"{test_metric_name} : {str(test_metric)}")


            # Save the pruned ontology to csv file.
            sub_dG = dG_prune.subgraph(
                    nx.shortest_path(dG_prune.to_undirected(), model.root))
            #onto_df = construct_ontology(
            #        sub_dG, model.root, title=f"Ontology\t\t{base_title}", 
            #        module_file=MODULE_FILE)
            #onto_df.to_csv(f"{RES_DIR}/Ontology_{base_savefile}.csv")
            
            # Save the pruned network-graph to csv file.
            network_df = construct_network(
                    model, sub_dG, classes, title=f"Network:\t{base_title}", 
                    module_file=MODULE_FILE)
            network_df.to_csv(f"{RES_DIR}/Network_{base_savefile}.csv")
            
            # Save the classification report (if CLASSIFICATION)
            if CLASSIFICATION:
                cm_analysis(
                        y_test, test_preds, f"{RES_DIR}/Class_{base_savefile}.png", 
                        labels=classes, title=base_title)
        

        # Assign scores to the ontology edges that have been removed.
        dropped_edges = set(dG_current.edges).difference(set(dG_prune.edges))
        if len(dropped_edges) != 0:
            for target, source in dropped_edges:
                ind = scores_df.index[(
                        (scores_df["Source"] == source) & (scores_df["Target"] == target))]
                ind = ind[0]
                scores_df.at[ind, "Score"] = score
            score += len(dropped_edges)    
            scores_df.to_csv(f"{RES_DIR}/Scores.csv")
        
        # Update the current state of the ontology.
        dG_current = dG_prune.copy()    
    
    
# Run the code.
if __name__ == "__main__":
    run_KnowledgeNet()



             
