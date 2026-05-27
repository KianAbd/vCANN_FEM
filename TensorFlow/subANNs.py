# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:10:58 2023

@author: Kian
"""

import tensorflow as tf

# @tf.keras.saving.register_keras_serializable()
class PositiveVarianceScaling(tf.keras.initializers.Initializer):
    
    def __init__(self, **kwargs):
        super(PositiveVarianceScaling, self).__init__(**kwargs)    
    
    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        weights = initializer(shape, dtype=dtype)
        return tf.abs(weights)
    

# @tf.keras.saving.register_keras_serializable()
class PositiveHeNormal(tf.keras.initializers.Initializer):
    
    def __init__(self, **kwargs):
        super(PositiveHeNormal, self).__init__(**kwargs)    
    
    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.HeNormal()
        weights = initializer(shape, dtype=dtype)
        return tf.abs(weights)

# @tf.keras.saving.register_keras_serializable()
class PositiveGlorotNormal(tf.keras.initializers.Initializer):
    
    def __init__(self, **kwargs):
        super(PositiveGlorotNormal, self).__init__(**kwargs)

    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.GlorotNormal()
        weights = initializer(shape, dtype=dtype)
        return tf.abs(weights)


# @tf.keras.saving.register_keras_serializable()
class NegativeHeNormal(tf.keras.initializers.Initializer):
    
    def __init__(self, **kwargs):
        super(NegativeHeNormal, self).__init__(**kwargs)    
    
    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.HeNormal()
        weights = initializer(shape, dtype=dtype)
        return -tf.abs(weights)
    
class NegConstraint(tf.keras.constraints.Constraint):
    """Constrains the weights to be negative."""
    
    def __call__(self, w):
        w = tf.keras.backend.convert_to_tensor(w)
        return w * tf.cast(tf.less_equal(w, 0.0), dtype=w.dtype)

def squared_softplus(x):
    # softplus(x) = log(1 + exp(x))
    return tf.math.softplus(x) ** 2

def neg_squared_softplus(x):
    # softplus(x) = log(1 + exp(x))
    return -tf.math.softplus(x) ** 2

from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'squared_softplus': squared_softplus})
get_custom_objects().update({'neg_squared_softplus': neg_squared_softplus})


def Psi_subANN(Invars, layer_size, activations, suffix):
    """
    Feed-forward neural network representing the strain energy function

    Parameters
    ----------
    Invars_in : tensor
        The generalized invariants.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.

    Returns
    -------
    x : tensor
        the output of the network, i.e. the strain energy.

    """
    x = Invars
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(
            size,
            activation=activations[i],
            kernel_initializer=PositiveVarianceScaling(),
            kernel_constraint=tf.keras.constraints.NonNeg(),
            use_bias=True,
            bias_initializer=None,
            # kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="Psi_{:}_{:}".format(suffix, str(i+1))
            )(x)
       
    x = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=PositiveVarianceScaling(),
        kernel_constraint=tf.keras.constraints.NonNeg(),
        use_bias=False,    # if set to True, kormos will not work
        bias_initializer=None,
        # kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="Psi_{:}_{:}".format(suffix, str(i+2))
        )(x)
    
    return x

#
###
#

def Psi_a_subANN(Invars, layer_size, activations, suffix):
    """
    Feed-forward neural network representing the potential from which the non-equilibirum
    strain energy function is constructed

    Parameters
    ----------
    Invars_in : tensor
        The generalized invariants.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.

    Returns
    -------
    x : tensor
        the output of the network, i.e. the strain energy.

    """
    x = Invars
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(
            size,
            activation=activations[i],
            kernel_initializer = PositiveVarianceScaling(), #  tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), # PositiveVarianceScaling(),
            kernel_constraint=tf.keras.constraints.NonNeg(),
            use_bias=True,
            bias_initializer=None,
            kernel_regularizer=tf.keras.regularizers.L2(0.0025),
            name="Psi_a_{:}_{:}".format(suffix, str(i+1))
            )(x)
        
    x = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer = PositiveVarianceScaling(), #  tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), # PositiveVarianceScaling(),
        kernel_constraint=tf.keras.constraints.NonNeg(),
        use_bias=False,   # if set to True, kormos will not work
        bias_initializer=None,
        kernel_regularizer=tf.keras.regularizers.L2(0.0025),
        name="Psi_a_{:}_{:}".format(suffix, str(i+2))
        )(x)
    return x

#
###
#


def Tau_subANN(Invars, layer_size, activations, suffix):
    """
    Feed-forward neural network representing the relaxation time
    (before scaling with the time constant equally spaced on the logarithmic scale.

    Parameters
    ----------
    Invars_in : tensor
        The generalized invariants.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.

    Returns
    -------
    x : tensor
        the output of the network, i.e. the relaxation time before scaling.

    """
    x = Invars
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(
            size,                             
            activation=activations[i],
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), #tf.keras.initializers.HeNormal(),
            bias_initializer=None,
            use_bias=True,
            # kernel_constraint=tf.keras.constraints.NonNeg(),
            # bias_constraint=tf.keras.constraints.NonNeg(),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            # bias_regularizer=tf.keras.regularizers.L2(0.001),
            name="Tau_{:}_{:}".format(suffix, str(i+1))
            )(x)
        
    if len(layer_size) == 0:
        i = -1   

    x = tf.keras.layers.Dense(
        1,
        activation=tf.keras.activations.softplus, # tf.math.abs, # ensure positivity of the relaxation times
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), #tf.keras.initializers.HeNormal(),
        bias_initializer=None,
        use_bias=True,
        # kernel_constraint=tf.keras.constraints.NonNeg(),
        # bias_constraint=tf.keras.constraints.NonNeg(),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        # bias_regularizer=tf.keras.regularizers.L2(0.001),
        name="Tau_{:}_{:}".format(suffix, str(i+2))
        )(x)
    return x

#
###
#

def G_subANN(Invars, layer_size, activations, suffix):
    """
    Feed-forward neural network representing the relaxation coefficient
    (before normalization such that the sum of all relaxation coefficients equals 1.)

    Parameters
    ----------
    Invars : tensor
        The generalized invariants.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.

    Returns
    -------
    x : tensor
        the output of the network, i.e. the relaxation coefficients before normalization.

    """
    x = Invars
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(size,
                                  activation=activations[i],
                                  kernel_initializer= tf.keras.initializers.HeNormal(),
                                  bias_initializer=None,
                                  use_bias=True,
                                  # kernel_constraint=tf.keras.constraints.NonNeg(),
                                  # bias_constraint=tf.keras.constraints.NonNeg(),
                                   kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                  name="G_{:}_{:}".format(suffix, str(i+1)))(x)
    if len(layer_size) == 0:
        i = -1
    x = tf.keras.layers.Dense(1,
                              activation=tf.keras.activations.softplus, # ensure positivity of the relaxation coefficients
                              kernel_initializer=tf.keras.initializers.HeNormal,
                              bias_initializer=None,
                              use_bias=True,
                              # kernel_constraint=tf.keras.constraints.NonNeg(),
                              # bias_constraint=tf.keras.constraints.NonNeg(),
                               kernel_regularizer=tf.keras.regularizers.L2(0.01),
                              name="G_{:}_{:}".format(suffix, str(i+2)))(x)
    return x


#
###
#

def summation(w, numDir):
    """
    Transforms the flat tensor that results from the weighting subANN to a properly shaped tensor which is normalized (the weight in each grou sum to 1).
    
    Parameters
    ----------
    w : tensor [shape: (?, nSteps,  numDir+1)]
        Flat tensor which results from the weighting factor subANN layers.
  
    Returns
    -------
    w : tensor [shape: (?, nSteps, numDir+1)]
        Reshaped tensor that represents the weighting factors in normalized form (the weight in each grou sum to 1).
        
    """

    sum = tf.norm(w, ord=1, axis=-1)
    shaper = tf.constant([1,1,0]) + (numDir+1)*tf.constant([0,0,1])
    w = tf.divide(w, tf.tile(tf.expand_dims(sum,-1), shaper))
    
    return w


def w_subANN(extra_struc, layer_size, activations, numDir, suffix):
    """
    Feed-forward neural network representing the weighting factors w of a
    generalized structural tensor.
    
    Parameters
    ----------
    extra_struc : tensor, shape (?, nSteps, numExtraStruc)
        feature vector from which the weights are learned.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.
    
    Returns
    -------
    w : tensor, shape = (?,nSteps,numDir+1)
        the weights of a generalized structural tensor.
    
    """
    x = extra_struc
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(size,
                                  activation=activations[i],
                                  kernel_initializer= None,
                                  bias_initializer=None,
                                  # kernel_constraint=tf.keras.constraints.NonNeg(),
                                  # bias_constraint=tf.keras.constraints.NonNeg(),
                                  name="w_{:}_{:}".format(suffix, str(i+1)))(x)
    if len(layer_size) == 0:
        i = -1
    x = tf.keras.layers.Dense(numDir+1,
                              activation=tf.keras.activations.softplus, # ensure positivity of the relaxation coefficients
                              kernel_initializer=None,
                              bias_initializer=None,
                              use_bias=True,
                              # kernel_constraint=tf.keras.constraints.NonNeg(),
                              # bias_constraint=tf.keras.constraints.NonNeg(),
                              name="w_{:}_{:}".format(suffix, str(i+2)))(x)
    
    w = tf.keras.layers.Lambda(lambda x: summation(x[0], x[1]), name='unitSum_'+suffix)([x, numDir])

    return w


def unitVector(dir):
    """
    Transforms the flat tensor that results from the directional subANN to a properly shaped tensor which is normalized (each vector is of length 1).

    Parameters
    ----------
    dir : tensor [shape: (?, nSteps, 3)]
        Flat tensor which results from the directional subANN layers.

    Returns
    -------
    dir : tensor [shape: (?, nSteps, 3)]
        Reshaped tensor that represents the directional vectors in normalized form (each direction vector is a unit vector).
        
    """
    
    length = tf.norm(dir, ord='euclidean', axis=-1)
    dir = tf.divide(dir, tf.tile(tf.expand_dims(length,-1), tf.constant([1,1,3])))
    
    return dir



def dir_subANN(extra_struc, layer_size, activations, suffix):
    """
    Feed-forward neural network representing a preferred material direction.
    
    Parameters
    ----------
    extra_struc : tensor, shape (?, nSteps, numExtraStruc,)
        feature vector from which the weights are learned.
    layer_size : list of int
        the number of neurons in each layer.
    activations : list of str or list of functions, e.g. tf.nn.softplus
        the activation function of each hidden layer.
    suffix : str
        used to uniquely name the layers.
    
    Returns
    -------
    dir : tensor, shape = (?,nSteps,3)
        a preferred material directions.
        
    """
    
    x = extra_struc
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(size,
                                  activation=activations[i],
                                  kernel_initializer= None,
                                  bias_initializer=None,
                                  # kernel_constraint=tf.keras.constraints.NonNeg(),
                                  # bias_constraint=tf.keras.constraints.NonNeg(),
                                  name="dir_{:}_{:}".format(suffix, str(i+1)))(x)
    if len(layer_size) == 0:
        i = -1
    x = tf.keras.layers.Dense(3,
                              activation=tf.keras.activations.tanh,
                              kernel_initializer=None,
                              bias_initializer=None,
                              use_bias=True,
                              # kernel_constraint=tf.keras.constraints.NonNeg(),
                              # bias_constraint=tf.keras.constraints.NonNeg(),
                              name="dir_{:}_{:}".format(suffix, str(i+2)))(x)
    
    dir = tf.keras.layers.Lambda(lambda x: unitVector(x), name='unitVector_'+suffix)(x)
        
    return dir # (?, nSteps, 3)
       


###############################################################################
###                                 Damage                                  ###
###############################################################################


def NetZeta(input_shape, layer_size, activations, suffix='0'):
    
    NetZeta_layers = list()

    for i, size in enumerate(layer_size):
        
        if i == len(layer_size)-1:
            bias_constraint =  tf.keras.constraints.NonNeg()
            kernel_constraint =  tf.keras.constraints.NonNeg()
        else:
            bias_constraint = None
            kernel_constraint = None
            
        NetZeta_layers.append(tf.keras.layers.Dense(size,
                                                    activation=activations[i],
                                                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                    bias_initializer=None,
                                                    #kernel_constraint=kernel_constraint,
                                                    #bias_constraint=bias_constraint,
                                                    #kernel_regularizer=tf.keras.regularizers.l2(l2=0.001,),
                                                    name="NetZeta_{:}_layer_{:}".format(suffix, str(i))))
    if len(layer_size) == 0:
        i = -1   
    # activations of second last layers are sigmoid, such that with NonNeg_Norm
    # constraint the damage variable is always between 0 and 1 
    NetZeta_layers.append(tf.keras.layers.Dense(1,
                                                activation='linear',
                                                use_bias=True,
                                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                #kernel_regularizer=tf.keras.regularizers.l2(l2=0.001,),
                                                #kernel_constraint=NonNeg_Norm(1.0),
                                                name="NetZeta_{:}_layer_out".format(suffix)))
    
    def NetZeta_instance(input):
        output = input
        for layer in NetZeta_layers:
            output = layer(output)
        return output
    
    return NetZeta_instance

#
###
#

def Gamma_subANN(Invars_in, layer_size, activations, suffix):
    
    x = Invars_in
    for i, size in enumerate(layer_size):
        x = tf.keras.layers.Dense(size,
                                  activation=activations[i],
                                  kernel_initializer=tf.keras.initializers.HeNormal(), # PositiveHeNormal(),
                                  kernel_constraint=None, #tf.keras.constraints.NonNeg(),
                                  use_bias=True,
                                  bias_initializer=None,
                                  #kernel_regularizer=tf.keras.regularizers.L2(0.001),
                                  name="Gamma_{:}_{:}".format(suffix, str(i+1)))(x)
        
        
    x = tf.keras.layers.Dense(1,
                              activation='linear',
                              kernel_initializer=tf.keras.initializers.HeNormal(), # PositiveHeNormal(),
                              kernel_constraint=None, # tf.keras.constraints.NonNeg(),
                              use_bias=True,
                              bias_initializer=None,
                              #kernel_regularizer=tf.keras.regularizers.L2(0.001),
                              name="Gamma_{:}_{:}".format(suffix, str(i+2)))(x)
    return x