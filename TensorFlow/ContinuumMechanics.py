# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:52:27 2023

@author: Kian
"""

# Standard import
import tensorflow as tf
import numpy as np

# Precision
tf.keras.backend.set_floatx('float64')
tf_float = 'float64'

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian


##########################################################################################
##########################################################################################
###### BASIC CONTINUUM MECHANICS FUNCTIONS ###############################################
##########################################################################################
##########################################################################################

# @tf.keras.saving.register_keras_serializable()
class MooneyRivlinLayer(tf.keras.layers.Layer):
    """
    Implements the computation of second Piola-Kirchhoff stress-like quantities
    """
    def __init__(self, c1, c2, **kwargs):
        super(MooneyRivlinLayer, self).__init__(**kwargs)
        self.kernel = tf.Variable([c1, c2], trainable=True, dtype=tf.float64,
                                  constraint=tf.keras.constraints.NonNeg()
                                  )
        self.use_bias = False
        
    def call(self, Invars):
        c1 = self.kernel[0]
        c2 = self.kernel[1]
        
        I = Invars[:,:,0:1]
        J = Invars[:,:,1:2]
        Psi = c1*(I-1) + c2*(J-1)
        
        return Psi
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(MooneyRivlinLayer, self).get_config()
        return config
    
    
class TauLayer(tf.keras.layers.Layer):
    """
    Implements the computation of second Piola-Kirchhoff stress-like quantities
    """
    def __init__(self, tau, **kwargs):
        super(TauLayer, self).__init__(**kwargs)
        self.tau = tau
        self.kernel = tf.Variable(tau, trainable=True, dtype=tf.float64,
                                  constraint=tf.keras.constraints.NonNeg()
                                  )
        self.use_bias = False
        
    def call(self, Invars):
        return tf.ones_like(Invars)[:,:,0:1]*self.kernel
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(TauLayer, self).get_config()
        return config

#
###
#

class IdenticalPolymerLayer(tf.keras.layers.Layer):
    """Scaled energy contribution with trainable coefficient beta."""

    def __init__(self, nMaxwell, suffix, **kwargs):
        super(IdenticalPolymerLayer, self).__init__(**kwargs)
        self.nMaxwell = nMaxwell
        self.kernel = self.add_weight(
            name='beta_'+suffix,
            shape=(3,),
            initializer=tf.keras.initializers.ones(),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            dtype=tf.float64,
        )

    def call(self, S_infy):

        S_infy = tf.expand_dims(S_infy,2)
        shaper = self.nMaxwell*tf.constant([0,0,3,0,0]) + tf.constant([1,1,0,1,1])
        S_infy = tf.tile(S_infy, shaper) # (?, nSteps, nMaxwell, 3, 3)

        batchSize = tf.shape(S_infy)[0]
        nSteps = tf.shape(S_infy)[1]
        betas = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.kernel, 0), 0), -1), -1)
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,3,3])
        betas = tf.tile(betas, shaper)

        return S_infy * betas

    def get_config(self):
        config = super(IdenticalPolymerLayer, self).get_config()
        return config

#
###
#

## Pre-defined fiber directions and weights

class dirModelSymFiber(tf.keras.models.Model):
    """
    Implements a model representing the symmetrically in-plane distributed preferred
    fiber directions based on a trainable fiber angle 'theta' withouth relying on a feature vector
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        """
        Initializes the attributes of the fiber direction model

        Parameters
        ----------
        numTens : int
            the number of generalized structural tensors.
        numDir : int
            the number of preferred material directions. Has to be an even number.

        Raises
        ------
        ValueError
            Raises an error if numDir is odd, since it is assumed that two fiber
            families are always symmetrically distributed including the same angle 'theta'.

        Returns
        -------
        None.

        """
        super(dirModelSymFiber, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = numDir
        if numDir % 2 != 0:
            raise ValueError('Only an even number of preferred material directions is allowed, since they are assumed to occure as symmetric pairs!')
        theta_init = tf.keras.initializers.Constant(value=np.pi/4.)
        self.theta = self.add_weight(shape=(numDir//2),
                                     initializer=theta_init,
                                     trainable=True,
                                     name='theta')
               
    def call(self, inputs):
        """
        computes the preferred material directions

        Parameters
        ----------
        inputs : tf.Tensor
            input data, for example the deformation gradient.
            Only needed to derive the batch size and number of time steps

        Returns
        -------
        dirs : tf.Tensor
            the preferred material directions

        """
        # fiber orientations
        # theta = tf.nn.sigmoid(self.theta)*np.pi/2.0 # works not so good better let take on arbitrary values
        sin = tf.sin(self.theta)
        cos = tf.cos(self.theta)
        zero = tf.zeros_like(self.theta, dtype=tf_float)

        # symmetric pairs of fibers
        l_1 = tf.stack([cos,  sin, zero], axis=-1)
        l_2 = tf.stack([cos, -sin, zero], axis=-1)
        
        # interleave the single fibers such that rows 2*i and 2*i+1 belong to the symmetric fiber pair i
        dirs = tf.reshape(tf.stack([l_1, l_2], axis=1), [-1, tf.shape(l_1)[1]], name='concat_dirs') # (numDir, 3)
       
        # expand dims: add batch and time step dimension and tile the fiber directions
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        dirs = tf.expand_dims(tf.expand_dims(dirs, 0), 0)
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        dirs = tf.tile(dirs, shaper, name='dirs')
        
        return dirs # (?, nSteps, numDirs, 3)
    
    
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(weightModelSymFiber, self).get_config()
        config.update(
            {
                "numTens": self.numTens,
                "numDir": self.numDir,
            }
        )
        return config
    

class weightModelSymFiber(tf.keras.models.Model):
    """
    Implements the trainable weights of the generalized structural tensor based on
    the symmetrically in-plane distributed preferred fiber directionss. 
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        """
        Initializes the weights of the generalized structural tensor
        
        Parameters
        ----------
        numTens : int
            the number of generalized structural tensors.
        numDir : int
            the number of preferred material directions. Has to be an even number.

        Returns
        -------
        None.

        """
        super(weightModelSymFiber, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = numDir
        w_init = tf.keras.initializers.Constant(value=0.0)
        self.w = self.add_weight(shape=(numTens,numDir+1),
                                 initializer=w_init,
                                 trainable=True,
                                 name='w')
        
    def call(self, inputs):
        
        # w = tf.math.sigmoid(self.w)
        w = tf.nn.softmax(self.w)
        
        # zero = tf.constant(0.0, dtype=tf_float)
        # one = tf.constant(1.0, dtype=tf_float)
        # # isotropic part
        # w_iso = tf.stack([one, zero, zero])
        # # anisotropic part, same weights for both fibers
        # w_ani1 = tf.stack([one-w, w, zero])
        # w_ani2 = tf.stack([one-w, zero, w])
        
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        # weights = tf.stack([w_iso, w_ani1, w_ani2], name='stack_weights')
        # weights = tf.stack([w_iso, w_ani1, w_ani2], name='stack_weights')
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        w = tf.expand_dims(tf.expand_dims(w, 0), 0)
        w = tf.tile(w, shaper, name='weights')

        return w


    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(dirModelSymFiber, self).get_config()
        config.update(
            {
                "numTens": self.numTens,
                "numDir": self.numDir,
            }
        )
        return config


class dirModelOrtho(tf.keras.models.Model):
    """
    Implements a model representing for orthotropic materials with prefered material
    directions aligned with the coordinate axes x and y. 
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        """
        Initializes the attributes of the fiber direction model

        Parameters
        ----------
        numTens : int
            the number of generalized structural tensors.

        Returns
        -------
        None.

        """
        super(dirModelOrtho, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = 3
               
    def call(self, inputs):
        """
        computes the preferred material directions

        Parameters
        ----------
        inputs : tf.Tensor
            input data, for example the deformation gradient.
            Only needed to derive the batch size and number of time steps

        Returns
        -------
        dirs : tf.Tensor
            the preferred material directions

        """

        # coordinate directions
        l_1 = tf.stack([1.0, 0.0, 0.0], axis=-1)
        l_2 = tf.stack([0.0, 1.0, 0.0], axis=-1)
        l_3 = tf.stack([0.0, 0.0, 1.0], axis=-1)
        
        # interleave the single fibers such that rows 2*i and 2*i+1 belong to the symmetric fiber pair i
        # dirs = tf.reshape(tf.stack([l_1, l_2, l_3], axis=0), [-1, tf.shape(l_1)[0]], name='concat_dirs') # (numDir, 3)
        dirs = tf.reshape(tf.stack([l_1, l_2, l_3], axis=0), [-1, tf.shape(l_1)[0]], name='concat_dirs') # (numDir, 3)
       
        # expand dims: add batch and time step dimension and tile the fiber directions
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        dirs = tf.expand_dims(tf.expand_dims(dirs, 0), 0)
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        dirs = tf.tile(dirs, shaper, name='dirs')
        
        return dirs # (?, nSteps, numDirs, 3)
    
    
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(dirModelOrtho, self).get_config()
        config.update(
            {
                "numTens": self.numTens,
                "numDir": self.numDir,
            }
        )
        return config
    

class weightModelOrtho(tf.keras.models.Model):
    """
    Implements the trainable weights of the generalized structural tensor
    withouth relying on the feature vector. 
    """
    
    def __init__(self, numTens, numDir, **kwargs):
        super(weightModelOrtho, self).__init__(**kwargs)
        self.numTens = numTens
        self.numDir = 3
        w_init = tf.keras.initializers.Constant(value=0.0)
        self.w = self.add_weight(shape=(1,self.numDir),
                                 initializer=w_init,
                                 trainable=True,
                                 name='w')
        
    def call(self, inputs):
        
        
        w = tf.nn.softmax(self.w)
        # w_ = (tf.ones_like(w) - w)/2

        # w = tf.constant([[0.55, 0.27, 0.18]], dtype=tf_float) # for testing purposes only
        # w_ = (tf.ones_like(w) - w)/2

        # w = tf.concat([w, w_], axis=0) # (numTens, numDir)
               
        batchSize = tf.shape(inputs)[0]
        nSteps = tf.shape(inputs)[1]
        shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
        w = tf.expand_dims(tf.expand_dims(w, 0), 0)
        w = tf.tile(w, shaper, name='weights')

        return w


    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(weightModelOrtho, self).get_config()
        config.update(
            {
                "numTens": self.numTens,
                "numDir": self.numDir,
            }
        )
        return config


#
###
#

# @tf.function
def ten2_H(L, w, nSteps, numDir, numTens): # Generalized structural tensors: H_r = \sum_i w_ri * L_i, (?,nSteps,numTens,3,3)
    """
    Computes the generalized structural tensors from classical structural tensors and the
    corresponding scalar weights
    
    Parameters
    ----------
    L : tf.Tensor
        Classical structural tensors.
    w : tf.Tensor
        Scalar weights of the generalized structural tensors.
    nSteps : int
        Number of time steps.
    numDir : int
        Number of preferred directions to use (0 for isotropy, more than 0 for anisotropy).
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    H : tf.Tensor
        Generalized structural tensors.

    """
    batchSize = tf.shape(w)[0]

    # Create L_0 and add it to L
    shaper = batchSize*tf.constant([1,0,0,0,0]) + tf.constant([0,0,1,1,1]) + nSteps*tf.constant([0,1,0,0,0])
    L_0 = 1.0/3.0 * tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.eye(3, dtype=tf_float),0),0),0), shaper)
    
    L = tf.cond(numDir > 0, lambda: tf.concat([L_0, L], axis=2, name='concat_L0_L'), lambda: L_0)  
    # L = tf.concat([L_0, L], axis=2, name='concat_L0_L')

    # Expand L (to get one for each numTens)
    shaper = numTens*tf.constant([0,0,1,0,0,0]) + tf.constant([1,1,0,1,1,1])
    L = tf.tile(tf.expand_dims(L, 2), shaper) # (?,nSteps,numTens,numDir+1,3,3)

    # Expand w
    shaper = tf.constant([1,1,1,1,3])
    w = tf.tile(tf.expand_dims(w, 4), shaper) # (?,nSteps,numTens,numDir+1,3)
    shaper = tf.constant([1,1,1,1,1,3])
    w = tf.tile(tf.expand_dims(w, 5), shaper) # (?,nSteps,numTens,numDir+1,3,3)

    # Multiply L with weights
    L_weighted = tf.math.multiply(L, w) # (?,nSteps,numTens,numDir+1,3,3)

    # Sum them up for the corresponding H
    H = tf.math.reduce_sum(L_weighted, axis=3) # (?,nSteps,numTens,3,3)

    return H


def invariants_I(C, H, numTens): # Generalized invariants I: I_r = trace(C*H_r) [?,nSteps,numTens]
    """
    Computes the generalized invariant I corresponding to the individual structural tensors H.
    Can be used for the incompressible as well as nearly incompressible case since the incompressiblity
    constraint det(C)=1 does not have to be considered.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized structural tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    I : tf.Tensor
        Generalized invariant I corresponding to the individual generalized structural tensors.

    """
    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_tile = tf.tile(tf.expand_dims(C, 2), shaper)
    I = tf.linalg.trace(tf.matmul(C_tile,H))
    
    return I


    
# use the incompressible invariants
def invariants_J_incomp(C_bar, H, numTens): # Generalized polyconvex isochoric invariants J: J_r = trace( C_bar^{-T}*H_r) [?,nSteps,numTens] 
    """
    Computes the incompresssible generalized invariant J corresponding to the individual structural tensors H.
    The kinematic constraint det(C)=1 is explicitly taken into account in the formulaiton of the invarians.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized structural tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    J : tf.Tensor
        Incompressible generalized invariants J corresponding to the individual generalized structural tensors.

    """

    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_bar_tile = tf.tile(tf.expand_dims(C_bar, 2), shaper)
    invTransC_bar = tf.linalg.inv(tf.transpose(C_bar_tile, perm=[0, 1, 2, 4, 3]))
    matmul = tf.matmul(invTransC_bar, H)
    J = tf.linalg.trace(matmul)
    
    return J


def invariants_J_comp(C, H, numTens): # Generalized invariants J: J_r = trace(cofactor(C)*H_r) [?,nSteps,numTens]
    """
    Computes the compresssible generalized invariant J corresponding to the individual structural tensors H.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
    H : tf.Tensor
        Scalar weights of the generalized structural tensors.
    numTens : int
        Number of generalized structural tensors to use (at least 1).
        
    Returns
    -------
    J : tf.Tensor
        Compressible generalized invariants J corresponding to the individual generalized structural tensors.

    """       
    
    shaper = tf.constant([1,1,0,1,1]) + numTens*tf.constant([0,0,1,0,0])
    C_tile = tf.tile(tf.expand_dims(C, 2), shaper)
    
    detC_tile = tf.linalg.det(C_tile)
    shaper = tf.constant([1,1,1,3])
    detC_tile = tf.tile(tf.expand_dims(detC_tile, 3), shaper)
    shaper = tf.constant([1,1,1,1,3])
    detC_tile = tf.tile(tf.expand_dims(detC_tile, 4), shaper)
    
    invTransC = tf.linalg.inv(tf.transpose(C_tile, perm=[0, 1, 2, 4, 3]))
    
    mul = tf.math.multiply(detC_tile, invTransC)
    matmul = tf.matmul(mul, H)
    J = tf.linalg.trace(matmul)
    
    return J

#
###
#

def ten2_L(dir): # Structural tensor L_i = l_i (x) l_i , shape = (?, nSteps, numDir, 3, 3)
    """
    Computes the classical structural tensors L = l \dyadic l.
    
    Parameters
    ----------
    dir : tf.Tensor
        Preferred material directions.
        
    Returns
    -------
    L : tf.Tensor
        Generalized structural tensors.

    """ 
    
    dir = tf.expand_dims(dir, -1) # (?, nSteps, numDir, 3, 1)
    dir_t = tf.transpose(dir, perm=[0, 1, 2, 4, 3]) # (?, nSteps, numDir, 1, 3)
    L = tf.linalg.matmul(dir, dir_t) # (?, nSteps, numDir, 3, 3)
    
    return L


def invariant_I3(C): # Third invariant of a tensor C: I3 = det(C) [?,nSteps,1]
    """
    Compute third invariant (determinant) of a tensor.
    
    Parameters
    ----------
    C : tf.Tensor
        Arbitrary 2nd-order tensor.
        
    Returns
    -------
    det_C : tf.Tensor
        The determinant of C.

    """     
    det_C = tf.expand_dims(tf.linalg.det(C), 2)
    return det_C

def ten2_C(F): # Right Cauchy-Green tensor: C = F^T * F [?,nSteps,3,3]
    """
    Compute the right Cauchy-Green deformation tensor from the deformation gradient.
    
    Parameters
    ----------
    dir : tf.Tensor
        Preferred material directions.
        
    Returns
    -------
    L : tf.Tensor
        Generalized structural tensors.

    """ 
    return tf.linalg.matmul(F,F,transpose_a=True)

def ten2_C_bar(C):
    """
    Compute the isochoric right Cauchy-Green deformation tensor.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
        
    Returns
    -------
    C_bar : tf.Tensor
        Isochoric right Cauchy-Green deformation tensors.

    """ 
    det_C = tf.linalg.det(C)
    scale = tf.math.pow(det_C, -1./3.)
    
    shaper = tf.constant([1,1,3])
    scale = tf.tile(tf.expand_dims(scale, 2), shaper)
    shaper = tf.constant([1,1,1,3])
    scale = tf.tile(tf.expand_dims(scale, 3), shaper)
    
    C_bar = tf.math.multiply(scale, C)
    
    return C_bar

   

def ten2_C_dot(F, F_dot): # material time derivative of the right Cauchy-Green tensor [?,nSteps,3,3]
    """
    Compute the material time derivative of the right Cauchy-Green deformation gradient.
    
    Parameters
    ----------
    F : tf.Tensor
        deformation gradient.
    F_dot : tf.Tensor
        material time derivative of the deformation gradient.
        
    Returns
    -------
    C_dot : tf.Tensor
        Material time derivative of the right Cauchy-Green deformation tensor.

    """ 
    C_dot = tf.linalg.matmul(F_dot,F,transpose_a=True) + tf.linalg.matmul(F,F_dot,transpose_a=True)
    return C_dot



def ten2_F_ref(F): # Deformation gradient in reference configuration [?,nSteps,3,3]
    """
    Compute the deformation gradient in the reference configuration (identity matrix).
    
    Parameters
    ----------
    F : tf.Tensor
        deformation gradient.
   
    Returns
    -------
    F_ref : tf.Tensor
        Referential deformation gradient.

    """ 
    # In Order for the other formulae to work we need the correct dimension required to produce enough eye matrices/tensors 
    batchSize = tf.shape(F)[0]
    nSteps = tf.shape(F)[1]
    shaper = batchSize*tf.constant([1,0,0,0]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])

    F_ref = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(3, dtype=tf_float),0),0), shaper)
    
    return F_ref


def grad(Psi, C):     
    dPsidC = tf.gradients(Psi, C, unconnected_gradients='zero')[0]
    return tf.math.scalar_mul(2.0, dPsidC)

def ten2_S_incomp(S_dev, C):
    """
    Compute incompressible 2. Piola-Kirchhoff stress tensor.
    
    Parameters
    ----------
    S_dev : tf.Tensor
        Deviatoric 2. Piola-Kirchhoff stress tensor.
    C : tf.Tensor
        right Cauchy-Green deformation tensor.        
   
    Returns
    -------
    F_ref : tf.Tensor
        Referential deformation gradient.

    """ 
    CInv = tf.linalg.inv(C)

    S_33 = tf.expand_dims(S_dev[:,:,2,2],2)
    CInv_33 = tf.expand_dims(CInv[:,:,2,2],2)
    p = tf.expand_dims(tf.divide(S_33, CInv_33),2) # hydrostatic pressure
    shaper = tf.constant([1,1,3,3])
    p = tf.tile(p, shaper)
    
    S_iso = tf.math.multiply(p, CInv)
    
    return tf.subtract(S_dev, S_iso)


def psi_vol(det_C): # volumetric strain energy contribution (c.f. Hartmann/Neff 2003, Diss Ebbig)
    """
    Volumetric strain energy contribution (c.f. Hartmann/Neff 2003, Diss Ebbig).
    
    Parameters
    ----------
    det_C : tf.Tensor
        Determinant of the right Cauchy-Green deformation tensor.
        
    Returns
    -------
    U : tf.Tensor
        Volumetric strain energy.

    """     
    J = tf.math.sqrt(det_C)
    U = J + 1./J - 2.
    return U


def ten2_E(C): # Green-Lagrange strain tensor: E=1/2*(C-I) [?,nsteps,3,3]
    """
    Compute the Green-Lagrange strain tensor from the right Cauchy-Green deformation tensor.
    
    Parameters
    ----------
    C : tf.Tensor
        Right Cauchy-Green deformation tensor.
        
    Returns
    -------
    E : tf.Tensor
        Green-Lagrange deformation tensors.

    """ 
    batchSize = tf.shape(C)[0]
    nSteps = tf.shape(C)[1]
    I = tf.expand_dims(tf.expand_dims(tf.eye(3, dtype=tf_float),0),0)
    shaper = batchSize*tf.constant([1,0,0,0,]) + nSteps*tf.constant([0,1,0,0]) + tf.constant([0,0,1,1])
    I = tf.tile(I, shaper)
    E = tf.math.scalar_mul(0.5, tf.math.subtract(C, I))
    return E


#
###
#

# @tf.keras.saving.register_keras_serializable()
class GradientLayer(tf.keras.layers.Layer):
    """
    Implements the computation of second Piola-Kirchhoff stress-like quantities
    """
    def __init__(self, scale=False, **kwargs):
        super(GradientLayer, self).__init__(**kwargs)
        self.scale = scale
        
    def call(self, y, x):
        dydx = tf.gradients(y, x, unconnected_gradients='zero')[0]
        if self.scale == True:
            dydx = tf.math.scalar_mul(2.0, dydx)
        return dydx
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(GradientLayer, self).get_config()

        config.update(
            {
                "scale": self.scale,
            }
        )
        return config

#
###
#

class BatchJacobianLayer(tf.keras.layers.Layer):
    """
    Implements the computation of second Piola-Kirchhoff stress-like quantities
    """
    def __init__(self, scale=False, **kwargs):
        super(BatchJacobianLayer, self).__init__(**kwargs)
        self.scale = scale
        
    def call(self, y, x):
        jac = batch_jacobian(y, x)
        if self.scale == True:
            jac = tf.math.scalar_mul(2.0, jac)
        return jac
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(BatchJacobianLayer, self).get_config()
        return config

#
###
#


class PsiSigmaLayer(tf.keras.layers.Layer):
    """"
    Implements for each generalized structural tensor the stress normalization strain energy contribution to guarantee a stress-free reference configuration
    """
    def __init__(self, **kwargs):
        super(PsiSigmaLayer, self).__init__(**kwargs)
        
    def call(self, alpha, beta, I, J):
        """
        Computes the strain energy contribution to guarantee a stress-free reference configuration
        
        Parameters
        ----------
        alpha : tf.Tensor
            factor depending on the partial derivatives of Psi evluated in the reference configuration. Either alpha or beta is zero.
        beta : tf.Tensor
            factor depending on the partial derivatives of Psi evluated in the reference configuration. Either alpha or beta is zero.
        I : tf.Tensor
            first generalized invariant of the corresponding generalized structural tensor.
        J : tf.Tensor
            secodn generalized invariant of the corresponding generalized structural tensor.

        Returns
        -------
        PsiSigma : tf.tensor
            stess normalization contribution.
        """
        
        Psi_1 = tf.math.multiply(alpha, tf.math.subtract(I, 1.0))
        Psi_2 = tf.math.multiply(beta, tf.math.subtract(J, 1.0))
        PsiSigma = tf.math.add(Psi_1, Psi_2)
    
        return PsiSigma
        
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(PsiSigmaLayer, self).get_config()
        return config

#
###
#

# @tf.keras.saving.register_keras_serializable()
class stressUpdateLayer(tf.keras.layers.Layer):
    """
    Implements the recursive stress update formula for computing the viscoelastic stress
    
    """
    
    def __init__(self, nMaxwell, nSteps, **kwargs):
        super(stressUpdateLayer, self).__init__(**kwargs)
        self.nMaxwell = nMaxwell
        self.nSteps = nSteps
   
    def call(self, S_e, t, PronyParams):
        """
        Computes the viscoelastic 2. Piola-Kirchhoff stress tensor.
        
        Parameters
        ----------
        S_e : tf.Tensor
            Instantaneous elastic 2. Piola-Kirchhoff stress tensor
        t : tf.Tensor
            Time
        PronyParams : tf.Tensor
            Prony Parameters; relaxation times and coefficients
                                
        Returns
        -------
        S : tf.Tensor
            Viscoelastic 2. Piola-Kirchhoff stress tensor
        S_infy : tf.Tensor
            Equlibrium 2. Piola-Kirchhoff stress tensor
        Q_sum : tf.Tensor
            Viscous 2. Piola-Kirchhoff stress tensor (sum of all Maxwell elements' viscous stress)
        """
         
        batchSize = tf.shape(S_e)[0] 
       
        tau = (PronyParams[:,:,:self.nMaxwell])
        g = (PronyParams[:,:,self.nMaxwell:])  
        g_sum = tf.math.reduce_sum(g, name='g_sum', axis=-1,keepdims=True)
        g = tf.divide(g, g_sum, name='g') # norm the sum of all g's to 1
        g_infy = g[:,:,0:1]     
        g_i = g[:,:,1:]     
        
        ###
        # Prony Series with variable relaxation times and coefficients
        def recursive_update(x0, x1):
            """
            Compute the stress update for one time step.
            
            """
            
            # x[0] - Q_i, viscoelastic overstresses
            # x[1] - t, time
            # x[2] - S_e, instantaneous elastic stress
            # x[3] - tau, relaxation times
            # x[4] - g, relaxation coefficients
            
            # average values over the time interval interval delta_t = x1[1] - x0[1]
            g_bar =  (x1[4] + x0[4])/2.
            tau_bar = (x1[3] + x0[3])/2.
            # compute overstress update
            delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, self.nMaxwell))
            
            term_1  = tf.math.exp(-delta_t/tau_bar)
            term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
            shaper  = tf.constant([1,1,3,3])
            term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
            
            term_2 = tf.math.exp(-delta_t/(2.0*tau_bar))*g_bar
            # term_2 = g_bar*tau_bar/ delta_t *(1.0 - tf.math.exp(-delta_t/tau_bar)) # alternative update rule
            term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
            shaper = tf.constant([1,1,3,3])
            term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
            
            delta_S_e = x1[2]-x0[2]
            delta_S_e = tf.expand_dims(delta_S_e,1)
            shaper = tf.constant([1,self.nMaxwell,1,1])
            delta_S_e = tf.tile(delta_S_e, shaper) # (?,nMaxwell,3,3)
            
            Q_i = tf.math.add(tf.math.multiply(term_1, x0[0]), tf.math.multiply(term_2, delta_S_e)) # (?,nMaxwell,3,3)
            
            result = (Q_i, x1[1], x1[2], x1[3], x1[4])
                   
            return result
    
    
        Q_zeros = tf.zeros([batchSize, self.nSteps, self.nMaxwell,3,3], dtype='float64')
        
        # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
        Q_zeros = tf.transpose(Q_zeros, perm=[1,0,2,3,4])
        S_e     = tf.transpose(S_e,     perm=[1,0,2,3])
        tau     = tf.transpose(tau,     perm=[1,0,2])
        g_i     = tf.transpose(g_i,     perm=[1,0,2])
        t       = tf.transpose(t,       perm=[1,0])
        
        initializer = (tf.zeros([batchSize, self.nMaxwell, 3, 3], dtype='float64'), -t[1], tf.zeros([batchSize, 3, 3], dtype='float64'), tau[0], g_i[0])
        
        # recursively compute the stress
        Q = tf.scan(
                recursive_update, # fn
                (Q_zeros, t, S_e, tau, g_i), #  elems
                initializer,
                parallel_iterations=10,
                name='Q'
            )
        
        Q_terms = Q[0]
        Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_Q_i', keepdims=False) # accumulate the Maxwell element contributions
        
        # transpose back to (?,nSteps,3,3) such that the 0-th dimension is again the batch and not the time steps
        S_e = tf.transpose(S_e, perm=[1,0,2,3]) # instantaneous elasic stress
        Q_sum = tf.transpose(Q_sum, perm=[1,0,2,3]) # viscous overstress
        
        g_infy = tf.expand_dims(g_infy,-1)
        shaper = tf.constant([1,1,3,3])
        g_infy = tf.tile(g_infy, shaper) # (?,nSteps,3,3)
        
        S_infy = tf.math.multiply(g_infy, S_e)
        S = S_infy + Q_sum
    
        ### comment in for pure relaxation test
        # Q_terms = g_i * tf.math.exp(-tf.reshape(time, [-1,1])/tau)
        # Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_h_i', keepdims=True)
        # stress = g_infy + Q_sum
        
        return S, S_infy, Q_sum
        
    
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(stressUpdateLayer, self).get_config()
        config.update(
            {
                'nMaxwell': self.nMaxwell,
                'nSteps': self.nSteps
            }
        )
        return config

#
###
#


def stressUpdate_Liu(inputs, nMaxwell, nSteps):

    S_a = inputs[0]
    t   = inputs[1]
    tau = inputs[2]
    
    batchSize = tf.shape(S_a)[0] 
      
    
    ###
    # Stress update
    def recursive_update(x0, x1):
        """
        Compute the stress update for one time step.
        
        """
        
        # x[0] - Q_i, viscoelastic overstresses
        # x[1] - t, time
        # x[2] - S_a, fictitious elastic stress
        # x[3] - tau, relaxation times
        
        # average values over the time interval interval delta_t = x1[1] - x0[1]
        tau = (x1[3]+x0[3])/2
        
        # compute overstress update
        delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, nMaxwell))
        
        term_1  = tf.math.exp(-delta_t/tau)
        term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
        shaper  = tf.constant([1,1,3,3])
        term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
        
        term_2 = tf.math.exp(-delta_t/(2.0*tau))
        term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
        shaper = tf.constant([1,1,3,3])
        term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
        
        delta_S_a = x1[2]-x0[2]
        
        Q_i = tf.math.add(tf.math.multiply(term_1, x0[0]), tf.math.multiply(term_2, delta_S_a)) # (?,nMaxwell,3,3)
        
        result = (Q_i, x1[1], x1[2], x1[3])
               
        return result


    Q_zeros = tf.zeros([batchSize, nSteps, nMaxwell, 3, 3], dtype='float64')
    
    # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
    Q_zeros = tf.transpose(Q_zeros, perm=[1,0,2,3,4])
    S_a     = tf.transpose(S_a,     perm=[1,0,2,3,4])
    tau     = tf.transpose(tau,     perm=[1,0,2])
    t       = tf.transpose(t,       perm=[1,0])
    
    initializer = (tf.zeros([batchSize, nMaxwell, 3, 3], dtype='float64'), -t[1], tf.zeros([batchSize, nMaxwell, 3, 3], dtype='float64'), tau[0])
    
    # recursively compute the stress
    Q = tf.scan(
            recursive_update, # fn
            (Q_zeros, t, S_a, tau), #  elems
            initializer,
            parallel_iterations=10,
            name='Q'
        )
    
    # transpose back to (?,nSteps,nMaxwell,3,3) such that the 0-th dimension is again the batch and not the time steps
    Q_ra = tf.transpose(Q[0], perm=[1,0,2,3,4]) # viscous overstress
    
    return Q_ra



# @tf.keras.saving.register_keras_serializable()
class stressUpdateLayer_Liu(tf.keras.layers.Layer):
    """
    Implements the recursive stress update formula for computing the viscoelastic stress
    
    """
    
    def __init__(self, nMaxwell, nSteps, **kwargs):
        super(stressUpdateLayer_Liu, self).__init__(**kwargs)
        self.nMaxwell = nMaxwell
        self.nSteps = nSteps
   
    def call(self, S_a, t, tau):
        """
        Computes the viscoelastic 2. Piola-Kirchhoff stress tensor.
        
        Parameters
        ----------
        S_a : tf.Tensor
            Fictitious 2. Piola-Kirchhoff stress tensor
        t : tf.Tensor
            Time
        tau : tf.Tensor
            relaxation times
                                
        Returns
        -------
        Q_sum : tf.Tensor
            Viscous 2. Piola-Kirchhoff stress tensor (sum of all Maxwell elements' viscous stress)
        """
         
        batchSize = tf.shape(S_a)[0] 
          
        
        ###
        # Stress update
        def recursive_update(x0, x1):
            """
            Compute the stress update for one time step.
            
            """
            
            # x[0] - Q_i, viscoelastic overstresses
            # x[1] - t, time
            # x[2] - S_a, fictitious elastic stress
            # x[3] - tau, relaxation times
            
            # average values over the time interval interval delta_t = x1[1] - x0[1]
            tau = x1[3]
            
            # compute overstress update
            delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, self.nMaxwell))
            
            term_1  = tf.math.exp(-delta_t/tau)
            term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
            shaper  = tf.constant([1,1,3,3])
            term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
            
            term_2 = tf.math.exp(-delta_t/(2.0*tau))
            term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
            shaper = tf.constant([1,1,3,3])
            term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
            
            delta_S_a = x1[2]-x0[2]
            
            Q_i = tf.math.add(tf.math.multiply(term_1, x0[0]), tf.math.multiply(term_2, delta_S_a)) # (?,nMaxwell,3,3)
            
            result = (Q_i, x1[1], x1[2], x1[3])
                   
            return result
    
    
        Q_zeros = tf.zeros([batchSize, self.nSteps, self.nMaxwell, 3, 3], dtype='float64')
        
        # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
        Q_zeros = tf.transpose(Q_zeros, perm=[1,0,2,3,4])
        S_a     = tf.transpose(S_a,     perm=[1,0,2,3,4])
        tau     = tf.transpose(tau,     perm=[1,0,2])
        t       = tf.transpose(t,       perm=[1,0])
        
        initializer = (tf.zeros([batchSize, self.nMaxwell, 3, 3], dtype='float64'), -t[1], tf.zeros([batchSize, self.nMaxwell, 3, 3], dtype='float64'), tau[0])
        
        # recursively compute the stress
        Q = tf.scan(
                recursive_update, # fn
                (Q_zeros, t, S_a, tau), #  elems
                initializer,
                parallel_iterations=10,
                name='Q'
            )
        
        # transpose back to (?,nSteps,nMaxwell,3,3) such that the 0-th dimension is again the batch and not the time steps
        Q_ra = tf.transpose(Q[0], perm=[1,0,2,3,4]) # viscous overstress
        
        return Q_ra
        
    
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(stressUpdateLayer, self).get_config()
        config.update(
            {
                'nMaxwell': self.nMaxwell,
                'nSteps': self.nSteps
            }
        )
        return config

#
###
#

def compute_S_a_neq_bar(dPsi_a_dJ, ddPsi_a_ddI, ddPsi_a_ddJ, ddPsi_a_dIdJ, C, Q, L):
    """
    
    Parameters
    ----------
    dPsi_a_dJ : Tensor, (?, nSteps, nMaxwell)
        first partial derivatives of Psi_a with respect to I.
    ddPsi_a_ddI : Tensor, (?, nSteps, nMaxwell)
        second partial derivatives of Psi_a with respect to I.
    ddPsi_a_ddJ : Tensor, (?, nSteps, nMaxwell)
        second partial derivatives of Psi_a with respect to J.
    ddPsi_a_dIdJ : Tensor, (?, nSteps, nMaxwell)
        second mixed partial derivatives of Psi_a with respect to I and J.
    C : Tensor, (?, nSteps, 3, 3)
        isochoric right Cauchy-Green deformation tensor.
    Q : Tensor, (?, nSteps, nMaxwell, 3, 3)
        Internal variable.
    L : Tensor, (?, nSteps, 3, 3)
        Generalized structural tensor, corresponding to the current structural tensor

    Returns
    -------
    None.

    """
    nMaxwell = tf.shape(Q)[2]

    ###
    shaper = nMaxwell*tf.constant([0,0,1,0,0]) + tf.constant([1,1,0,1,1])
    L = tf.tile(tf.expand_dims(L, 2), shaper) # (?,nsteps,nMaxwell,3,3)
    C = tf.tile(tf.expand_dims(C, 2), shaper) # (?,nsteps,nMaxwell,3,3)
    
    Cinv = tf.linalg.inv(C) # (?,nsteps,nMaxwell,3,3)
    H = tf.linalg.matmul(tf.linalg.matmul(Cinv,L),Cinv) # (?,nsteps,nMaxwell,3,3)

    CinvQH = tf.linalg.matmul(tf.linalg.matmul(Cinv, Q), H) # (?,nsteps,nMaxwell,3,3)
    HQCinv = tf.linalg.matmul(tf.linalg.matmul(H, Q), Cinv) # (?,nsteps,nMaxwell,3,3)
    
    ###
    shaper = tf.constant([1,1,1,3,3])

    QL = tf.linalg.matmul(Q,L) # (?,nsteps,nMaxwell,3,3)  
    trQL = tf.expand_dims(tf.expand_dims(tf.linalg.trace(QL), -1), -1) # (?,nsteps,nMaxwell,1,1)  
    trQL = tf.tile(trQL, shaper) # (?,nsteps,nMaxwell,3,3)
    
    QH = tf.linalg.matmul(Q,H) # (?,nsteps,nMaxwell,3,3)
    trQH = tf.expand_dims(tf.expand_dims(tf.linalg.trace(QH), -1), -1) # (?,nsteps,nMaxwell,1,1)  
    trQH = tf.tile(trQH, shaper) # (?,nsteps,nMaxwell,3,3)

    QCinv = tf.linalg.matmul(Q,Cinv) # (?,nsteps,nMaxwell,3,3)
    trQCinv = tf.expand_dims(tf.expand_dims(tf.linalg.trace(QCinv), -1), -1) # (?,nsteps,nMaxwell,1,1)  
    trQCinv = tf.tile(trQCinv, shaper) # (?,nsteps,nMaxwell,3,3)
    
    ###
    shaper = tf.constant([1,1,1,3,3])

    dPsi_a_dJ    = tf.expand_dims(tf.expand_dims(dPsi_a_dJ,    -1), -1) # (?,nsteps,nMaxwell,1,1) 
    ddPsi_a_ddI  = tf.expand_dims(tf.expand_dims(ddPsi_a_ddI,  -1), -1) # (?,nsteps,nMaxwell,1,1) 
    ddPsi_a_ddJ  = tf.expand_dims(tf.expand_dims(ddPsi_a_ddJ,  -1), -1) # (?,nsteps,nMaxwell,1,1) 
    ddPsi_a_dIdJ = tf.expand_dims(tf.expand_dims(ddPsi_a_dIdJ, -1), -1) # (?,nsteps,nMaxwell,1,1) 
    
    dPsi_a_dJ    = tf.tile(dPsi_a_dJ,    shaper) # (?,nsteps,nMaxwell,3,3)
    ddPsi_a_ddI  = tf.tile(ddPsi_a_ddI,  shaper) # (?,nsteps,nMaxwell,3,3)
    ddPsi_a_ddJ  = tf.tile(ddPsi_a_ddJ,  shaper) # (?,nsteps,nMaxwell,3,3)
    ddPsi_a_dIdJ = tf.tile(ddPsi_a_dIdJ, shaper) # (?,nsteps,nMaxwell,3,3)
    
    
    ###
    term_1 = tf.multiply(ddPsi_a_ddI, trQL)
    term_1 = tf.multiply(term_1, L)
    
    term_2 = tf.multiply(ddPsi_a_ddJ, trQH)
    term_2 = tf.multiply(term_2, H)
    
    term_3 = tf.add(CinvQH, HQCinv)
    term_3 = tf.multiply(dPsi_a_dJ, term_3)
    
    term_4 = tf.add(tf.multiply(trQH , L), tf.multiply(trQL, H)) 
    term_4 = tf.multiply(ddPsi_a_dIdJ, term_4)
    
    S_a_neq_bar = 2.0*(term_1 + term_2 + term_3 - term_4)   
        
    return S_a_neq_bar
    
    
#
###
#

def compute_hessian(Psi, Invars):
        
    hess = tf.hessians(Psi, Invars)
    return hess


class TangentLayer(tf.keras.layers.Layer):

    
    def __init__(self, nMaxwell, nSteps, **kwargs):
        super(TangentLayer, self).__init__(**kwargs)
        self.nMaxwell = nMaxwell
        self.nSteps = nSteps
        
    # @tf.function    
    def call(self, S_a, C_bar):
        
        dS_a_dC = [] # tf.TensorArray(tf.float64, size=self.nMaxwell)
        # dS_a_dC = tf.TensorArray(tf.float64, size=self.nMaxwell)
        for mm in range(self.nMaxwell):
            grads = []
            # grads = tf.TensorArray(tf.float64, size=9)
            for ii in range(3):
                for jj in range(3):
                    g = tf.gradients(S_a[:,:,mm,ii:ii+1,jj:jj+1], C_bar, unconnected_gradients='zero')[0] # (?,nSteps,3,3)
                    # g = tf.divide(g, 2.0)
                    
                    grads.append(g)
                    # cnt = ii*3 + jj
                    # grads = grads.write(cnt, g)
            
            # grads = tf.transpose(grads.stack(), (1,2,0,3,4))        
            grads = tf.stack(grads, axis=2) # (?,nSteps,9,3,3)
           
            gradients = tf.reshape(grads, (-1,self.nSteps,3,3,3,3)) # (?,nSteps,3,3,3,3) #???
            gradients_t = tf.transpose(gradients, perm=(0,1,3,2,4,5))
            gradients_sym = (gradients+gradients_t)/2
            
            # dS_a_dC.write(mm, gradients_sym)  
            # dS_a_dC = tf.unstack(dS_a_dC.stack(), num=self.nMaxwell, axis=0)
            dS_a_dC.append(gradients_sym)
            
        return dS_a_dC

        
        
    def get_config(self):
        # Implement get_config to enable serialization.
        config = super(TangentLayer, self).get_config()
        config.update(
            {
                'nMaxwell': self.nMaxwell,
                'nSteps': self.nSteps
            }
        )
        return config
    


@tf.function
def tangent(S_a, C_bar, nMaxwell, nSteps, batchSize):

    nMaxwell  = tf.shape(S_a)[2]
    nSteps    = tf.shape(S_a)[1]
    batch_size = tf.shape(S_a)[0]
    
    dS_a_dC = [] # tf.TensorArray(tf.float64, size=self.nMaxwell)
    for mm in range(3):
        # grads = []
        grads = [] # tf.TensorArray(tf.float64, size=9)
        for ii in range(3):
            for jj in range(3):
                g = tf.keras.layers.Lambda( lambda x: grad(x[0], x[1]))([S_a[:,:,mm,ii:ii+1,jj:jj+1], C_bar]) # (?,nSteps,3,3)
                g = tf.divide(g, 2.0)
                
                grads.append(g)
                # cnt = ii*3 + jj
                # grads = grads.write(cnt, g)
        
        # grads = tf.transpose(grads.stack(), (1,2,0,3,4))        
        
        grads = tf.stack(grads, axis=2) # (?,nSteps,9,3,3)
        gradients = tf.reshape(grads, (-1,nSteps,3,3,3,3)) # (?,nSteps,3,3,3,3) #???
        gradients_t = tf.transpose(gradients, perm=(0,1,3,2,4,5))
        gradients_sym = tf.math.divide(tf.math.add(gradients, gradients_t), 2.0)
        
        # dS_a_dC.write(mm, gradients_sym)  
        # dS_a_dC = tf.unstack(dS_a_dC.stack(), num=self.nMaxwell, axis=0)
        dS_a_dC.append(gradients_sym)
        
    return dS_a_dC


def batch_jacobian_fn(S_a, C):
    
    batchSize = tf.shape(S_a)[0]
    nSteps    = tf.shape(S_a)[1]
    nMaxwell  = tf.shape(S_a)[2]
    
    
    # x[0] - dS_a_dC, tangent
    # x[1] - S_a, fictitious elastic stress
    # x[2] - C, right Cauchy-Green deformation tensor
    
    def jac_fn(x0, x1):
        
        nMaxwell  = tf.shape(x1[1])[1]

        # S_a_ = tf.reshape(x1[1], [-1, nMaxwell, 9], name='S_a_flattened')
        S_a_unstack = tf.unstack(x1[1], axis=1, name='unstack_S_a')
        dS_a_dC = []
        for mm, s in enumerate(S_a_unstack):

            grads = []
            for ii in range(3):
                for jj in range(3):
                    grad = tf.gradients(s[:,ii,jj], x1[2], unconnected_gradients='zero')[0]
                    grads.append(grad)
                    
            grads = tf.reshape(tf.stack(grads, axis=-1), shape=[-1,3,3,3,3])
            dS_a_dC.append(grads)
        
        dS_a_dC = tf.stack(dS_a_dC, axis=1)
        # dS_a_dC = [ batch_jacobian(s, x1[2]) for s in S_a_unstack ]
        # dS_a_dC = tf.con
        
        result = (dS_a_dC, x1[1], x1[2])
        
        return result
    
    
    dS_a_dC_zeros = tf.zeros([batchSize, nSteps, nMaxwell, 3, 3, 3, 3], dtype=tf_float)    
    
    # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
    dS_a_dC_zeros = tf.transpose(dS_a_dC_zeros, perm=[1,0,2,3,4,5,6])
    S_a           = tf.transpose(S_a,           perm=[1,0,2,3,4])    
    C             = tf.transpose(C,             perm=[1,0,2,3])    

    initializer = (tf.zeros([batchSize, nMaxwell, 3, 3, 3, 3], dtype=tf_float), S_a[0], C[0])

    result = tf.scan(
                jac_fn, # fn
                (dS_a_dC_zeros, S_a, C), #  elems
                initializer,
                parallel_iterations=10,
                name='dS_a_dC'
                )
    
    dS_a_dC = result[0]
    dS_a_dC = tf.transpose(dS_a_dC, perm=[1,0,2,3,4,5,6])
    
    return dS_a_dC

#
###
#

def deviatoric_projection(S_bar, C):
    """
    Computes the deviatoric projection of the fictious (stress) tensor S_bar.

    Parameters
    ----------
    S_bar : Tensor, shape=(None,nSteps,3,3)
        fictitious (stress) tensor.
    C : Tensor, shape=(None,nSteps,3,3)
        Right Cauchy-Green deformation tensor.

    Returns
    -------
    S_dev : Tensor, shape=(None,nSteps,3,3)
        The deviatoric (stress) tensor.

    """
    
    trSC = tf.linalg.trace(tf.linalg.matmul(S_bar, C))
    trSC = tf.expand_dims(tf.expand_dims(trSC, -1), -1)
    shaper = tf.constant([1,1,3,3])
    trSC = tf.tile(trSC, shaper)
    
    Cinv = tf.linalg.inv(C)
    
    S_dev = tf.math.subtract(S_bar, tf.math.multiply(tf.math.divide(trSC,3), Cinv))
    
    return S_dev

#
###
#

def deviatoric_projection_transposed(S, C):
    """
    Computes projection of S with the transposed deviatoric projection tensor.

    Parameters
    ----------
    S : Tensor, shape=(None,nSteps,3,3)
        fictitious (stress) tensor.
    C : Tensor, shape=(None,nSteps,3,3)
        Right Cauchy-Green deformation tensor.

    Returns
    -------
    S_dev : Tensor, shape=(None,nSteps,3,3)
        The deviatoric (stress) tensor.

    """
    det_C = tf.linalg.det(C)
    scale = tf.math.pow(det_C, -1./3.)

    scale = tf.expand_dims(tf.expand_dims(scale, -1), -1)
    shaper = tf.constant([1,1,1,3])
    scale = tf.tile(scale, shaper)

    Cinv = tf.linalg.inv(C)
    
    trSCinv = tf.linalg.trace(tf.linalg.matmul(S, Cinv))
    trSCinv = tf.expand_dims(tf.expand_dims(trSCinv, -1), -1)
    shaper = tf.constant([1,1,3,3])
    trSCinv = tf.tile(trSCinv, shaper)

    S_dev = tf.multiply(scale, tf.math.subtract(S, tf.math.multiply(tf.math.divide(trSCinv,3), C)))

    return S_dev
#
###
#

def stressUpdate(inputs, nMaxwell, nSteps):
    """
    Computes the viscoelastic 2. Piola-Kirchhoff stress tensor.
    
    Parameters
    ----------
    inputs : list of tf.Tensor
        [S_e, t, PronyParams], where:
            S_e : tf.Tensor
                Instantaneous elastic 2. Piola-Kirchhoff stress tensor
            t : tf.Tensor
                Time
            PronyParams : tf.Tensor
                Prony Parameters; relaxation times and coefficients
                
    nMaxwell : int
        Number of Maxwell elements.
    nSteps : int
        Number of time steps
        
    Returns
    -------
    S : tf.Tensor
        Viscoelastic 2. Piola-Kirchhoff stress tensor
    S_infy : tf.Tensor
        Equlibrium 2. Piola-Kirchhoff stress tensor
    Q_sum : tf.Tensor
        Viscous 2. Piola-Kirchhoff stress tensor (sum of all Maxwell elements' viscous stress)
    """
    
    S_e = inputs[0]
    t = inputs[1]
    PronyParams = inputs[2]
    
    batchSize = tf.shape(S_e)[0] 
   
    tau = (PronyParams[:,:,:nMaxwell])
    g = (PronyParams[:,:,nMaxwell:])  
    g_sum = tf.math.reduce_sum(g, name='g_sum', axis=-1,keepdims=True)
    g = tf.divide(g, g_sum, name='g') # norm the sum of all g's to 1
    g_infy = g[:,:,0:1]     
    g_i = g[:,:,1:]     
    
    ###
    # Prony Series with variable relaxation times and coefficients
    def prony_fn(x0, x1):
        
        # x[0] - Q_i, viscoelastic overstresses
        # x[1] - t, time
        # x[2] - S_e, instantaneous elastic stress
        # x[3] - tau, relaxation times
        # x[4] - g, relaxation coefficients
        
        # average values over the time interval interval delta_t = x1[1] - x0[1]
        g_bar =  (x1[4] + x0[4])/2.
        tau_bar = (x1[3] + x0[3])/2.
        # compute overstress update
        delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, nMaxwell))
        
        term_1  = tf.math.exp(-delta_t/tau_bar)
        term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
        shaper  = tf.constant([1,1,3,3])
        term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
        
        term_2 = tf.math.exp(-delta_t/(2.0*tau_bar))*g_bar
        # term_2 = g_bar*tau_bar/ delta_t *(1.0 - tf.math.exp(-delta_t/tau_bar)) # alternative update rule
        term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
        shaper = tf.constant([1,1,3,3])
        term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
        
        delta_S_e = x1[2]-x0[2]
        delta_S_e = tf.expand_dims(delta_S_e,1)
        shaper = tf.constant([1,nMaxwell,1,1])
        delta_S_e = tf.tile(delta_S_e, shaper) # (?,nMaxwell,3,3)
        
        Q_i = tf.math.add(tf.math.multiply(term_1, x0[0]), tf.math.multiply(term_2, delta_S_e)) # (?,nMaxwell,3,3)
        
        result = (Q_i, x1[1], x1[2], x1[3], x1[4])
               
        return result

    Q_zeros = tf.zeros([batchSize, nSteps, nMaxwell,3,3], dtype=tf_float)
    
    # transpose to scan over the 0-th dimension which is should be the time steps and not the batches
    Q_zeros = tf.transpose(Q_zeros, perm=[1,0,2,3,4])
    S_e     = tf.transpose(S_e,     perm=[1,0,2,3])
    tau     = tf.transpose(tau,     perm=[1,0,2])
    g_i     = tf.transpose(g_i,     perm=[1,0,2])
    t       = tf.transpose(t,       perm=[1,0])
    
    initializer = (tf.zeros([batchSize, nMaxwell, 3, 3], dtype=tf_float), -t[1], tf.zeros([batchSize, 3, 3], dtype=tf_float), tau[0], g_i[0])
    
    Q = tf.scan(
            prony_fn, # fn
            (Q_zeros, t, S_e, tau, g_i), #  elems
            initializer,
            parallel_iterations=10,
            name='Q'
        )
    
    Q_terms = Q[0]
    Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_Q_i', keepdims=False) # accumulate the Maxwell element contributions
    
    # transpose back to (?,nSteps,3,3) such that the 0-th dimension is again the batch and not the time steps
    S_e = tf.transpose(S_e, perm=[1,0,2,3]) # instantaneous elasic stress
    Q_sum = tf.transpose(Q_sum, perm=[1,0,2,3]) # viscous overstress
    
    g_infy = tf.expand_dims(g_infy,-1)
    shaper = tf.constant([1,1,3,3])
    g_infy = tf.tile(g_infy, shaper) # (?,nSteps,3,3)
    
    S_infy = tf.math.multiply(g_infy, S_e)
    S = S_infy + Q_sum

    ### pure relaxation test
    # Q_terms = g_i * tf.math.exp(-tf.reshape(time, [-1,1])/tau)
    # Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_h_i', keepdims=True)
    # stress = g_infy + Q_sum
    
    return S, S_infy, Q_sum



def ten2_P(S, F): # Second Piola Kirchhoff stress tensor: P = F * S [?,nSteps,3,3]
    """
    Compute the 1. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    P : tf.Tensor
        1. Piola-Kirchhoff stress tensor.
        
    """
    P = tf.matmul(F, S)
    return P


def ten2_sigma(P, F, J): # Cauchy stress tensor: sigma = J^-1 * P * F^T [?,nSteps,3,3]
    """
    Compute the Cauchy stress by pushing forward the 2. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    sigma : tf.Tensor
        Cauchy stress tensor.
        
    """ 
    OneOverJ = tf.tile(tf.expand_dims(tf.math.divide(1.0,J),-1), tf.constant([1,1,3,3]))
    sigma = tf.math.multiply(OneOverJ, tf.matmul(P, tf.transpose(F, perm=[0, 1, 3, 2])))

    return sigma


def stressTensors(S, F):
    """
    Compute the 1. Piola-Kirchhoff stress and the Cauchy stress by pushing forward the 2. Piola-Kirchhoff stress.
    
    Parameters
    ----------
    S : tf.Tensor
        2. Piola-Kirchhoff stress tensor.
    F : tf.Tensor
        Deformation gradient.
        
    Returns
    -------
    P : tf.Tensor
        1. Piola-Kirchhoff stress tensor.
    P11 : tf.Tensor
        (1,1)-component of the 1. Piola-Kirchhoff stress tensor.
    sigma : tf.Tensor
        Cauchy stress tensor.
        
    """ 
    P     = tf.keras.layers.Lambda(lambda x: ten2_P(x[0], x[1]), name='P')([S, F])
    P11   = tf.keras.layers.Lambda(lambda P: P[:,:,0,0] , name='P11')(P)
    J     = tf.keras.layers.Lambda(lambda F: tf.expand_dims(tf.linalg.det(F),-1), name='J')(F)
    sigma = tf.keras.layers.Lambda(lambda x: ten2_sigma(x[0], x[1], x[2]), name='sigma')([P, F, J])
    
    return P, P11, sigma
    
    
