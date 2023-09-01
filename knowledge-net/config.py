import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import regularizers 



# General parameters
EXPERIMENT_NAME = "../experiments/Top20/KN"
EXP_DIR = f"../experiments/{EXPERIMENT_NAME}"
RES_DIR = f"{EXP_DIR}/results"
MODULE_FILE = "Data/modules.csv"
#MODULE_FILE = None
CLASSIFICATION = True
BUILD_MODE = False
TEST_SIZE = 0.2
INPUT_DIM = 20

# Saving parameters
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_LOADDIR = f"{EXP_DIR}/Models/Model_1"
MODEL_SAVEDIR = f"{EXP_DIR}/Models/Model_1"

# BUILD_MODE parameters (if BUILD_MODE is True)
FUNC = "x[0]**2 + sqrt(abs(x[1])) + 2*x[2] - 3*x[3] + x[8] * x[9]"
#FUNC = "x[1] + 2*x[2] - 3*x[3]"
NOISE_SD = "0"
DATA_SIZE = 1000
LOWER = -10
UPPER = 10


# PRODUCTION_MODE (aka BUILD_MODE is false)
DATA_FILE = f"{EXP_DIR}/data/data.tsv"

# Model parameters (Architecture)
###LOSS_FN = tf.keras.losses.MeanSquaredError()
###LOSS_FN = tf.keras.losses.BinaryCrossentropy()
LOSS_FN = tf.keras.losses.CategoricalCrossentropy()

OUTPUT_DIM = 5
OUTPUT_ACT = "softmax"
MODULE_ACT = "tanh"
INPUT_ACT = "linear"
AUX_ACT = "linear"
BATCHNORM = True
AUX = False
MODULE_NEURONS_FUNC = "256"
WEIGHTS_INIT = initializers.GlorotUniform()


# Model parameters (Training and Regularization)
TRAIN_EPOCHS = 100
BATCH_SIZE = 256
RETRAIN = True
RETRAIN_EPOCHS = 50
INPUT_REG = regularizers.L2(0.0005)
MODULE_REG = regularizers.L2(0.0005)
PRUNE_TRAIN_ITERATIONS = 20000
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

