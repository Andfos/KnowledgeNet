import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import regularizers 



# General parameters
EXPERIMENT_NAME = "Toy1"
EXP_DIR = f"Experiments/{EXPERIMENT_NAME}"
RES_DIR = f"{EXP_DIR}/Results"
#MODULE_FILE = f"{EXP_DIR}/Data/modules.csv"
MODULE_FILE = None
CLASSIFICATION = False
BUILD_MODE = True
TEST_SIZE = 0.2
INPUT_DIM = 11

# Saving parameters
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_LOADDIR = f"{EXP_DIR}/Models/Model_shap"
MODEL_SAVEDIR = f"{EXP_DIR}/Models/Model_shap"

# BUILD_MODE parameters (if BUILD_MODE is True)
FUNC = "x[0]**2 + sqrt(abs(x[1])) + 2*x[2] + 3*x[3] + x[8] * x[9]"
#FUNC = "x[1] + 2*x[2] + 3*x[3]"
NOISE_SD = "0"
DATA_SIZE = 1000
LOWER = -10
UPPER = 10


# PRODUCTION_MODE (aka BUILD_MODE is false)
DATA_FILE = f"{EXP_DIR}/Data/input_data.tsv"

# Model parameters (Architecture)
LOSS_FN = tf.keras.losses.MeanSquaredError()
###LOSS_FN = tf.keras.losses.BinaryCrossentropy()
#LOSS_FN = tf.keras.losses.CategoricalCrossentropy()

OUTPUT_DIM = 1
OUTPUT_ACT = "linear"
MODULE_ACT = "tanh"
INPUT_ACT = "linear"
AUX_ACT = "linear"
BATCHNORM = True
AUX = False
MODULE_NEURONS_FUNC = "64"
WEIGHTS_INIT = initializers.GlorotUniform()


# Model parameters (Training and Regularization)
TRAIN_EPOCHS = 15000
BATCH_SIZE = 800
RETRAIN = True
RETRAIN_EPOCHS = 4000
INPUT_REG = regularizers.L2(0.01)
MODULE_REG = regularizers.L2(0.01)
PRUNE_TRAIN_ITERATIONS = 20000
UPDATE_ITERS = 50

#regl0 = 0.001
#reg_glasso = 5
gl_pen = 0.0005
l0_pen = 0.0005
MAX_GL = 10
MAX_L0 = 10

SHUFFLE_BUFFER_SIZE = 200

#ngene = 11
