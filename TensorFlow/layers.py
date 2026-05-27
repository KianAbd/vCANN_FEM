# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:56:10 2025

@author: Kian
"""

import tensorflow as tf




# @tf.keras.saving.register_keras_serializable()
class ScaleLayer(tf.keras.layers.Layer):
    """
    Scales the relaxation times to powers of 10 within the predefined range [tau_min, tau_max]
    
    """
    
    def __init__(self, tau_min=-4, tau_max=4, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.tau_min = tau_min
        self.tau_max = tau_max
        
    def call(self, x):
        nPronyParams = x.shape[2]
        scale = tf.experimental.numpy.logspace(self.tau_min, self.tau_max, 
                                               num=nPronyParams, endpoint=True,
                                               base=10.0, axis=0) # dtype=tf_float,
        scale = tf.expand_dims(scale, 0)
        scaled = tf.math.multiply(x, scale)
        return scaled 

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(ScaleLayer, self).get_config()
        config.update(
            {
                "tau_min": self.tau_min,
                "tau_max": self.tau_max,
            }
        )
        return config

#
###
#


class GammaLayer(tf.keras.layers.Layer):
    """
    Returns the value of the damage function, depending on the value of Delta.
    Accounts for the fact, that during primary loading gamma has to be zero.
    
    """
    
    def __init__(self, lambda_Gamma, **kwargs):
        super(GammaLayer, self).__init__(**kwargs)
        self.lambda_Gamma = lambda_Gamma
        
    def __call__(self, Gamma, Delta, Psi_max):
        
        Gamma_ = tf.where(tf.math.logical_or(Delta > 0.0, Psi_max == 0.0, name='check_Psi_Delta'), Gamma, 
                          tf.zeros_like(Gamma, name='Gamma_primary'), name='correct_Gamma') # dtype=tf_float
        self.add_loss(self.lambda_Gamma*tf.reduce_sum(tf.nn.relu(-Gamma)))
        
        return Gamma_

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(GammaLayer, self).get_config()
        return config 


#
###
#


class MaxLayer(tf.keras.layers.Layer):
    """
    A layer that computes the maximum value of an array reached up to the current index
    
    """
    
    def __init__(self, **kwargs):
        super(MaxLayer, self).__init__(**kwargs)
        
    def __call__(self, x):
        
        def fn(x_old, x_new):
            return tf.math.maximum(x_old, x_new)
        
        # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
        x = tf.transpose(x, perm=[1,0,2])
        
        x_max = tf.scan(fn, x, name='scan_max')

        # undo transposition such that first and second dimension are batch and time again
        x_max = tf.transpose(x_max, perm=[1,0,2])
        
        return x_max

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(MaxLayer, self).get_config()
        return config    
#
###
#