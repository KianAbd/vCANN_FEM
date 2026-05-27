# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:57:24 2025

@author: Kian
"""

import tensorflow as tf
import os
import math

import Callbacks
import utils
import kormos

class MinMaxNormalizedLoss(tf.keras.losses.Loss):
    def __init__(self, name="normalized_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = y_true[:, :, 0, 0]
        y_pred = y_pred[:, :, 0, 0]
        y_min = tf.math.reduce_min(y_true, axis=-1, keepdims=True)
        y_max = tf.math.reduce_max(y_true, axis=-1, keepdims=True)
        # clamp range to avoid blow-up for near-constant samples
        y_range = tf.math.maximum(y_max - y_min, 1e-3)
        per_sample_loss = tf.math.reduce_mean(((y_true - y_pred) / y_range)**2, axis=-1)
        return tf.math.reduce_mean(per_sample_loss)

class IndividuallyNormalizedLoss(tf.keras.losses.Loss):
    def __init__(self, name="normalized_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        # weights = tf.ones((batch_size,), dtype=tf.float64)
        # weights = tf.tensor_scatter_nd_update(weights, [[0]], [1.0])  # weight for the first sample in the batch
        # W = tf.reshape(weights, (-1, 1))
        norm_y_true = ((y_true - y_pred) / (y_true + 1e-3))**2
        # norm_y_true = (y_true - y_pred)**2
        # norm_y_true_weighted = W * norm_y_true
        return tf.math.reduce_mean(norm_y_true)

class WeightedLoss(tf.keras.losses.Loss):
    def __init__(self, name="weighted_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = y_true[:, :, 0, 0]
        y_pred = y_pred[:, :, 0, 0]
        batch_size = tf.shape(y_true)[0]
        weights = tf.ones((batch_size,), dtype=tf.float64)
        weights = tf.tensor_scatter_nd_update(weights, [[0]], [0.1])  # weight for the first sample in the batch
        W = tf.reshape(weights, (-1, 1))
        diff = W * (y_true - y_pred)**2
        return tf.math.reduce_mean(diff)


    
def stochastic(model, trainDs, epochs, earlyStopPatience, folder, valDs=None, Maxwell_monitor=None, reg_callback=None, clr_config=None): 

    #tensorboard callback    
    log_dir = os.path.join(folder, "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                log_dir=log_dir,
                                                histogram_freq=0,
                                                write_graph=True,
                                                write_images=False,
                                                update_freq=50,
                                                profile_batch=0,
                                                embeddings_freq=0,
                                                embeddings_metadata=None,
                                                )

    
    # custom checkpoint callback to enable saving every n-th epoch  
    ckpt_dir = os.path.join(folder, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'ckpt-epoch-{epoch:02d}.ckpt')
    model_ckpt_callback = Callbacks.MyModelCheckpoint(epoch_per_save=20,
                                                      filepath=ckpt_path,
                                                      monitor='val_loss',
                                                      verbose=2,
                                                      save_best_only=False,
                                                      save_weights_only=True,
                                                      mode='auto',
                                                      options=None
                                                      )
        
    # early stopping based on the validation loss without regularization loss
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=earlyStopPatience,
                                                  restore_best_weights=True, # restore weights of best epoch
                                                  mode='auto')
    
    # callback to compute the validation loss without the sparsity enforcing L1 regularization
    val_loss_callback = Callbacks.ValLossCallback(validation_data=valDs)
    
    
    callbacks = [val_loss_callback, model_ckpt_callback, early_stop, tensorboard_callback]

    if reg_callback is not None:    
        # callback to write the regularization weights of the Maxwell elements to the console
        callbacks.append(reg_callback)

    if Maxwell_monitor is not None:
        callbacks.append(Maxwell_monitor)

    if clr_config is not None:
        base_lr = clr_config.get('base_lr', 1e-3)
        max_lr = clr_config.get('max_lr', clr_config.get('lr', base_lr * 100))
        step_size = clr_config.get('step_size', 20)  # measured in epochs

        def triangular_cyclic_lr(epoch, lr=None):
            cycle = math.floor(1 + epoch / (2 * step_size))
            x = abs(epoch / step_size - 2 * cycle + 1)
            scale = max(0.0, 1 - x)
            return base_lr + (max_lr - base_lr) * scale

        clr_callback = tf.keras.callbacks.LearningRateScheduler(triangular_cyclic_lr, verbose=0)
        callbacks.append(clr_callback)


    history = model.fit(
        x=trainDs,
        batch_size=None,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=valDs,
        shuffle=True,
        # validation_freq=1,
        max_queue_size=20,
        workers=8,
        use_multiprocessing=True
    )
    
    return history, model
 

#
###
#

def deterministic(model, trainDs, epochs, earlyStopPatience, folder, valDs=None, method='L-BFGS-B', Maxwell_monitor=None, loss=tf.keras.losses.MeanSquaredError(), batch_size=2**10):
    
    
    # define bounds / constraints
    bounds = utils.setBounds(model)
    constraints = utils.boundsAsConstraints(bounds)
    
    # set up model and optimizer
    kormos_model = kormos.models.BatchOptimizedModel(inputs=model.inputs, outputs=model.outputs)
    optimizer = kormos.optimizers.ScipyBatchOptimizer(method=method, bounds=bounds, batch_size=batch_size)            
    kormos_model.compile(loss=loss, optimizer=optimizer, metrics=['mean_squared_error'])
    options={'ftol': 1e-307, 'gtol': 1e-307} # options passed to the scipy minimizer
    
    # define callbacks
    callbacks = []
    
    ckpt_dir = os.path.join(folder, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'ckpt-epoch-{epoch:02d}.ckpt')
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                             save_freq=20,
                                                             monitor='val_loss',
                                                             verbose=2,
                                                             save_best_only=False,
                                                             save_weights_only=True,
                                                             mode="auto",
                                                             options=None,
                                                             )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=earlyStopPatience,
                                                  restore_best_weights=True, # restore weights of best epoch
                                                  mode='auto')
    
    callbacks.append(model_ckpt_callback)
    callbacks.append(early_stop)
    
    if Maxwell_monitor is not None:
        callbacks.append(Maxwell_monitor)

    # train the model using a deterministic optimizer
    with tf.device('/device:CPU:0'):
        history = kormos_model.fit(trainDs,
                                   epochs=epochs,
                                   callbacks=callbacks,
                                   batch_size=batch_size,
                                   validation_data=valDs,
                                   verbose=0,
                                   options=options
                                   )
     
    # check after training if bouds are violated   
    utils.checkBounds(model, bounds)
    
    return history, model