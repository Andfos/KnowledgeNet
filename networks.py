import tensorflow as tf
import keras.layers
from tensorflow import keras
from tensorflow.keras import initializers
import numpy as np
import keras.backend as K
from tensorflow.keras import layers
from keras.layers import Dense, Layer, Input, Concatenate, Add, BatchNormalization
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from utils import set_module_neurons
from tensorflow.keras import regularizers 
import pandas as pd


from keras.layers import Input









class RestrictedLayer(Dense):
    """ Build a layer where a specific connections matrix can be applied.  """
    
    def __init__(self, units, connections, pbias=None, **kwargs):
        """ 
        Initialize a RestrictedLayer that inherits from the Keras Dense
        class.

        Args:
            units(int): Number of nodes for layer
            connections(numpy.ndarray): Connections mask for weights 
                    leaving current layer.
            pbias(float): Bias to be added to the final output of the node 
                    following the activation.
        """

        super().__init__(units, **kwargs)
        self.connections = connections
        
        if pbias is not None:
            self.pbias = tf.Variable([pbias], name=f"{self.name}/pbias")

    def call(self, inputs):
        """ 
        Call the restricted layer on a set of inputs.

        Operation: output = activation(dot(input, kernel)+ bias) + pbias

        Args:
            inputs

        """
        
        output = K.dot(inputs, self.kernel * self.connections)

        # If the bias is to be included, add it with the output of the previous
        # step.
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        
        # Apply the activation function and then add the pbias term.
        output = self.activation(output)
        
        if hasattr(self, "pbias"):
            output = tf.math.add(self.pbias, output)
        
        return output

















class RestrictedNN(tf.keras.Model):
    """ 
    Creates a neural network where connections between modules can be specified.

    



    """
    
    def __init__(self,
                 output_dim,
                 output_act,
                 module_act,
                 input_act,
                 root, 
                 dG, 
                 input_dim, 
                 module_neurons_func, 
                 term_direct_input_map,
                 mod_size_map, 
                 initializer,
                 input_regularizer,
                 module_regularizer,
                 loss_fn,
                 aux=False, 
                 batchnorm=False):
        """
        Initialize a Restricted Neural Network

        Args:
            output_dim(int): Number of neurons to use in the output layer.
            root(str): Name of the root node of the ontology
            dG(nx.DiGraph): The directed graph representing the ontology
            input_dim(int): The dimensions of the input space.
            module_neurons_func(str): String defining the function used to 
                    allocate neurons to modules
            term_direct_input_map(dict):
            mod_size_map(dict): Dictionary defining number of inputs annotated 
                    to each term

        """



        super(RestrictedNN, self).__init__()

        
        self.output_dim = output_dim
        self.output_act = output_act
        self.module_act = module_act
        self.input_act = input_act
        self.root = root
        self.dG = dG       
        self.input_dim = input_dim
        self.module_neurons_func = module_neurons_func
        self.term_direct_input_map = term_direct_input_map
        self.mod_size_map = mod_size_map
        self.initializer = initializer
        self.input_regularizer = input_regularizer
        self.module_regularizer = module_regularizer
        self.loss_fn = loss_fn
        self.aux = aux
        self.batchnorm = batchnorm
        self.get_module_dimensions()
        
        # Construct the input layer.
        self.build_input_layer()
        
        # Construct the layers between modules.
        self.build_module_layers()
        



    def get_module_dimensions(self):
        """Retrieve the number of neurons allocated to  each  module layers.
         
        The number of neurons in each module layer is passed in as function.
        It can be static, where each module layer gets the same number of 
        inputs, or it can be dynamic, where the number of neurons is a 
        function of the number of inputs to the layer.
        """


        dG_copy = self.dG.copy()
        self.module_order = []
        self.module_dimensions = {}
        self.module_children_num = {}
        self.module_children = {}
        self.incoming_weights = {}        
        activations = {}
        aux_enabled = {}
        batchnorm_enabled = {}
        

        
        while True:
            leaves = [n for n,d in dG_copy.out_degree() if d==0]
            if len(leaves) == 0:
                break
            self.module_order += leaves
            dG_copy.remove_nodes_from(leaves)





        for mod in self.module_order:
            num_children = len(self.dG[mod])
            self.module_children[mod] = []
            [self.module_children[mod].append(c) for c in self.dG.neighbors(mod)]
            
            if mod == self.root:
                num_neurons = self.output_dim
                activations[mod] = self.output_act
                aux_enabled[mod] = False
                batchnorm_enabled[mod] = False
            elif num_children == 0:
                num_neurons = 1
                activations[mod] = self.input_act
                aux_enabled[mod] = False
                batchnorm_enabled[mod] = False
            else:
                num_neurons = set_module_neurons(
                        num_children, 
                        self.module_neurons_func)
                activations[mod] = self.module_act
                #modules.append(f"{mod}-aux") if self.aux else False
                aux_enabled[mod] = True if self.aux else False
                batchnorm_enabled[mod] = True if self.batchnorm else False


            self.module_children_num[mod] =  num_children 
            self.module_dimensions[mod] = num_neurons
        
        # Get the number of incoming weights for each module
        for mod in self.dG.nodes():
            if self.aux == True:
                self.incoming_weights[mod] = self.module_children_num[mod]
            else:
                iw = 0
                for child in self.module_children[mod]:
                    iw += self.module_dimensions[child]
                self.incoming_weights[mod] = iw


            


            
        self.network_dims = pd.DataFrame()
        self.network_dims["Module"] = self.module_order
        self.network_dims["Direct_children_num"] = self.network_dims["Module"].map(
                self.module_children_num)
        self.network_dims["Annotated_terms"] = self.network_dims["Module"].map(
                self.mod_size_map)
        self.network_dims["Neurons"] = self.network_dims["Module"].map(
                self.module_dimensions) 
        self.network_dims["Children"] = self.network_dims["Module"].map(
                self.module_children) 
        self.network_dims["Input_shape"] = self.network_dims["Module"].map(
                self.incoming_weights) 
        self.network_dims["Activation"] = self.network_dims["Module"].map(
                activations) 
        self.network_dims["Aux_enabled"] = self.network_dims["Module"].map(
                aux_enabled) 
        self.network_dims["Batchnorm"] = self.network_dims["Module"].map(
                batchnorm_enabled) 

        print(self.network_dims)



    def build_input_layer(self):
        """ Construct the input layer for each input-connected module.

        The name of the input layer follows the form <module_name>_inp. The
        layer is passed the entire vector of inputs, and it returns only the 
        inputs that belong to the specified module, with each input multiplied 
        by one weight. For instance, if module1 takes input from inp1 and inp2, 
        the name of the layer will be module1_inp, it will have 2 neurons, and 
        the output of neuron 1 will equal the value of input 1 multiplied by 
        weight1, and the output of neuron 2 will equal the value of input 2 
        multiplied by weight2. The weights are initialized to 0 and the 
        activation is linear.
        """
        

        self.network_layers = {}
        self.grad_masks = {}
        
        # Iterate through the modules that are directly mapped to inputs.
        for mod, input_set in self.term_direct_input_map.items():
            input_set = sorted(list(input_set))

            # Construct the connections matrix.
            connections = np.zeros((self.input_dim, len(input_set)))
            j = 0
            for id in input_set:
                connections[id][j] = 1
                j+=1
            
            # The gradient mask for the input layer will be initially the same
            # as the connections matrix.
            input_layer_name = f"{mod}_inp"

            # Initialize the RestrictedLayer with a kernel of 0's, then add it 
            #to the input_layers dictionary.
            self.network_layers[input_layer_name] = (RestrictedLayer(
                    units=len(input_set),
                    connections=connections,
                    input_shape=(self.input_dim, ),
                    activation="linear",
                    pbias = 1.0,
                    use_bias=False,
                    name=input_layer_name,
                kernel_initializer=initializers.Zeros(),
                kernel_regularizer=self.input_regularizer,
                trainable=True))



    def build_module_layers(self):
        """ Construct the layers for each module in the ontology.

        The module layer takes in a tensor output from a module_inp layer if 
        it is directly mapped to inputs, or it takes in a tensor that is
        created by concatenating the tensors output from all of the module's 
        child modules. By default, the sigmoid activation is applied and a bias 
        term is included. 
        """
        
        #self.module_layers = {}
        #self.mod_layer_list = []   
        self.module_dims = (
                self.network_dims[self.network_dims["Direct_children_num"] != 0])
        
        for index, row in self.module_dims.iterrows():
            mod = row["Module"]
            children = row["Children"]
            neurons = row["Neurons"]
            input_shape = row["Input_shape"]
            activation = row["Activation"]
            aux_enabled = row["Aux_enabled"]
            batchnorm_enabled = row["Batchnorm"]
            
            # Create the module layers.
            connections = np.ones((input_shape, neurons))
            self.network_layers[f"{mod}_mod"] = RestrictedLayer(
                    neurons, 
                    connections=connections,
                    activation=activation, 
                    name=f"{mod}_mod",
                    #pbias = 1.0,
                    use_bias=True,
                    kernel_initializer=self.initializer,
                    kernel_regularizer=self.module_regularizer)
            
            # Create the auxiliary layers (if AUX).
            if aux_enabled:
                connections = np.ones((neurons, 1))
                self.network_layers[f"{mod}_aux"] = RestrictedLayer(
                        1, 
                        connections,
                        input_shape=(neurons,),
                        activation="linear", 
                        name=f"{mod}_aux", 
                        use_bias=False, 
                        kernel_initializer=self.initializer)

            # Create the batchnorm layers (if BATCHNORM).
            if batchnorm_enabled:
                self.network_layers[f"{mod}_batchnorm"] = (
                        BatchNormalization(axis=1))



    def construct_model(self, batch_train = True):
        
        output_map = {}
        inputs = keras.Input(shape=(self.input_dim,))
        for i, row in self.module_dims.iterrows():
            print(row)
            mod = row["Module"]
            children = row["Children"]
            aux_enabled = row["Aux_enabled"]
            batchnorm = row["Batchnorm"]
            input_list = []
            
            # Pass input tensors through input layers.
            if mod in self.term_direct_input_map:
                layer = self.network_layers[f"{mod}_inp"]
                input_list.append(layer(inputs))
            
            # Concatenate the tensors that input to the module layer.
            for child_mod in children:
                if child_mod[0].islower():
                    continue
                input_list.append(output_map[child_mod])
            input = tf.concat(input_list, 1)
            
            # Pass the concatenated inputs through the module layer.
            layer = self.network_layers[f"{mod}_mod"]
            output = layer(input)
            
            # Batchnormalize the output (if BATCHNORM).
            if batchnorm:
                #output = tf.reshape(output, [-1, 32])
                #output = tf.reshape(output, output.shape)
                #print(output.shape)
                layer = self.network_layers[f"{mod}_batchnorm"]
                layer.trainable = batch_train
                output = layer(output)
            
            # Pass the output through an auxiliary layer (if AUX).
            if aux_enabled:
                layer = self.network_layers[f"{mod}_aux"]
                output = layer(output)

        
            # Keep the output of each module in output_map.
            output_map[mod] = output

        test_model = keras.Model(inputs=inputs, outputs=output)
        # Return the output from the root of the ontology.
        self.test_model = test_model





    
    def call(self, inputs, batch_train=True):
                
        output_map = {}
        

        # Iterate through the modules of the ontology.
        for i, row in self.module_dims.iterrows():
            mod = row["Module"]
            children = row["Children"]
            aux_enabled = row["Aux_enabled"]
            batchnorm = row["Batchnorm"]
            input_list = []
            
            # Pass input tensors through input layers.
            if mod in self.term_direct_input_map:
                layer = self.network_layers[f"{mod}_inp"]
                input_list.append(layer(inputs))
            
            # Concatenate the tensors that input to the module layer.
            for child_mod in children:
                if child_mod[0].islower():
                    continue
                input_list.append(output_map[child_mod])
            input = tf.concat(input_list, 1)
            
            # Pass the concatenated inputs through the module layer.
            layer = self.network_layers[f"{mod}_mod"]
            output = layer(input)
            
            # Batchnormalize the output (if BATCHNORM).
            if batchnorm:
                #output = tf.reshape(output, [-1, 32])
                #output = tf.reshape(output, output.shape)
                #print(output.shape)
                layer = self.network_layers[f"{mod}_batchnorm"]
                layer.trainable = batch_train
                output = layer(output)
            
            # Pass the output through an auxiliary layer (if AUX).
            if aux_enabled:
                layer = self.network_layers[f"{mod}_aux"]
                output = layer(output)
            
            # Keep the output of each module in output_map.
            output_map[mod] = output
        
        # Return the output from the root of the ontology.
        return(output_map[self.root])

            
        

        
        

            

            


def save_model(filepath, model, model_rmse, rmse_threshold = 1):
    """ Checks if model has converged to solution. Model is saved if it has."""
    converge = False
    if model_rmse <= rmse_threshold:
        print("Model saved.")
        converge = True
        model.save(filepath)

    return converge




















            



