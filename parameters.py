import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import regularizers 



# General parameters
EXPERIMENT_NAME = "Experiment_top20"
EXP_DIR = f"../Experiments/{EXPERIMENT_NAME}"
RES_DIR = f"{EXP_DIR}/Results_shap"
MODULE_FILE = f"{EXP_DIR}/Data/modules.csv"
CLASSIFICATION = True
BUILD_MODE = False
TEST_SIZE = 0.2
INPUT_DIM = 20

# Saving parameters
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_LOADDIR = f"{EXP_DIR}/Models/Model_shap"
MODEL_SAVEDIR = f"{EXP_DIR}/Models/Model_shap"

# BUILD_MODE parameters (if BUILD_MODE is True)
#FUNC = "0.2*x[0] * 2*x[1] + 2*x[2] + 3*x[3] + x[8] * x[9]"
FUNC = "x[1] + 2*x[2] + 3*x[3]"
NOISE_SD = "0"
DATA_SIZE = 1000
LOWER = -10
UPPER = 10


# PRODUCTION_MODE (aka BUILD_MODE is false)
DATA_FILE = f"{EXP_DIR}/Data/input_data.tsv"

# Model parameters (Architecture)
###LOSS_FN = tf.keras.losses.MeanSquaredError()
###LOSS_FN = tf.keras.losses.BinaryCrossentropy()
LOSS_FN = tf.keras.losses.CategoricalCrossentropy()

OUTPUT_DIM = 5
OUTPUT_ACT = "softmax"
MODULE_ACT = "tanh"
INPUT_ACT = "linear"
AUX_ACT = "linear"
BATCHNORM = False
AUX = False
MODULE_NEURONS_FUNC = "128"
WEIGHTS_INIT = initializers.GlorotUniform()


# Model parameters (Training and Regularization)
TRAIN_EPOCHS = 200
BATCH_SIZE = 256
RETRAIN = True
RETRAIN_EPOCHS = 100
INPUT_REG = regularizers.L2(0.001)
MODULE_REG = regularizers.L1(0.001)
PRUNE_TRAIN_ITERATIONS = 20000
UPDATE_ITERS = 20

#regl0 = 0.001
#reg_glasso = 5
gl_pen = 0.0005
l0_pen = 0.0005


SHUFFLE_BUFFER_SIZE = 200

#ngene = 11
