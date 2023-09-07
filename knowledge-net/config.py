"""
This configuration module is used for setting environment variables 
for a particular KnowledgeNet experiment. This module is imported into 
``main.py``, allowing the script to use the variables specified herein. 
Alternatively, this module can be imported into a jupyter notebook or 
user-created script.
"""

import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

"""
General Parameters
------------------

EXPERIMENT_NAME : str
    The name of the experiment. This should be the same name as the name of the
    parent directory that will host the data and results files of the
    experiment. Default is 'regression'.
MODULE_FILE : str
    This should specify the path to the file containing the relationships 
    between inputs and modules, or child modules and parent modules. Note: This
    is different than the ontology file.
CLASSIFICAION : bool
    Specify whether the experiment is one of classification. Default is `False`.
BUILD_MODE : bool
    Specify whether experiment data will be built on the fly from a user 
    defined function, rather than pulled from a data file.
TEST_SIZE : float
    Specify the ratio of test set size to training set size.
INPUT_DIM : int
    Specify the total number of features used by the model. 
"""

# General parameters
EXPERIMENT_NAME = "regression"
EXP_DIR = f"../tests/{EXPERIMENT_NAME}"
RES_DIR = f"{EXP_DIR}/results"
MODULE_FILE = "data/modules.csv"
#MODULE_FILE = None
CLASSIFICATION = False
BUILD_MODE = True
TEST_SIZE = 0.2
INPUT_DIM = 10


"""
Saving Parameters
-----------------
LOAD_MODEL : bool
    Specify whether the model will be loaded from a saved file. Default is
    `False`.
SAVE_MODEL : bool
    Specify whether the model should be saved. Default is
    `False`.
"""

# Saving parameters
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_LOADDIR = f"{EXP_DIR}/models/Model_1"
MODEL_SAVEDIR = f"{EXP_DIR}/models/Model_1"

# BUILD_MODE parameters (if BUILD_MODE is True)
FUNC = "x[0]**2 + 2*x[2] - 3*x[3] + x[8] * x[9]"
#FUNC = "x[1] + 2*x[2] - 3*x[3]"
NOISE_SD = "0"
DATA_SIZE = 1000
LOWER = -10
UPPER = 10


# PRODUCTION_MODE (aka BUILD_MODE is false)
DATA_FILE = f"{EXP_DIR}/data/data.tsv"

# Model parameters (Architecture)
LOSS_FN = tf.keras.losses.MeanSquaredError()
###LOSS_FN = tf.keras.losses.BinaryCrossentropy()
###LOSS_FN = tf.keras.losses.CategoricalCrossentropy()

OUTPUT_DIM = 1
OUTPUT_ACT = "linear"
MODULE_ACT = "tanh"
INPUT_ACT = "linear"
AUX_ACT = "linear"
BATCHNORM = True
AUX = False
MODULE_NEURONS_FUNC = "256"
WEIGHTS_INIT = initializers.GlorotUniform()


# Model parameters (Training and Regularization)
TRAIN_EPOCHS = 3000
BATCH_SIZE = 800
RETRAIN = True
RETRAIN_EPOCHS = 50
INPUT_REG = regularizers.L2(0.0005)
MODULE_REG = regularizers.L2(0.0005)
PRUNE_TRAIN_ITERATIONS = 2
UPDATE_ITERS = 20
SHUFFLE_BUFFER_SIZE = 200


# Model parameters (Pruning)
GL_PEN1 = 0.00005
L0_PEN1 = 0.0001

GL_PEN2 = 0.0001
L0_PEN2 = 0.0001

MAX_GL1 = 0.999
MAX_L01 = 0.999
MAX_GL2 = 0.999
MAX_L02 = 0.999

