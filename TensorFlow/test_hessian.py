# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:57:01 2025

@author: Kian
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Example shapes
batch_size = 2  # dynamic batch size
nSteps = 100

def fn(x):
    y = tf.math.reduce_sum(x[:,:,0:1]+2*x[:,:,1:2], axis=-1, keepdims=True)
    return tf.pow(y,3)

# Compute the Hessian

def compute_hessian(y, x):
    hess = tf.hessians(y,x)
    
    # nSteps = hess.shape[1]    
    # hess = [hess[:,ii,:,ii,::] for ii in range(nSteps)]
    # hess = tf.stack(hess, axis=1)
    
    return hess


# Define x and y as tensors
x = tf.tile(tf.Variable([[[0,1,], [2,3,],[4,5,]] ], dtype='float64'), (batch_size,nSteps,1))
y = fn(x)   # Replace with your function mapping x to y

# @tf.function
# def compute_hessian_(y, x):
#     with tf.GradientTape(persistent=True) as tape2:
#         tape2.watch(x)
#         with tf.GradientTape(persistent=True) as tape1:
#             tape1.watch(x)
#             y_out = fn(x)  # Your function y(x)
#         grad = tape1.gradient(y_out, x)  # First-order derivative
#     hessian = tape2.batch_jacobian(grad, x)  # Second-order derivative
    
#     nSteps = hessian.shape[1]
#     hessian = [hessian[:,ii,:,ii,::] for ii in range(nSteps)]
#     hessian = tf.stack(hessian, axis=1)
    
#     return hessian



inp = tf.keras.layers.Input((nSteps*3,2),batch_size=batch_size)
y   = tf.keras.layers.Lambda(lambda x: fn(x) )(inp)
out = tf.keras.layers.Lambda(lambda x: tf.hessians(x[0],x[1]) )([y,inp])

# out = tf.keras.layers.Lambda(lambda x: compute_hessian(x[0], x[1]))([y, inp])

model = tf.keras.models.Model(inp, out)
hess = model.predict(x,batch_size=batch_size)




def PronySeries_Liu(inputs, nMaxwell, nSteps):

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
        tau = x0[3]
        t_ = x1[1]
        # compute overstress update
        delta_t = tf.tile(tf.expand_dims(x1[1]-x0[1],-1), (1, nMaxwell))
        
        term_1  = tf.math.exp(-delta_t/tau)
        term_1  = tf.expand_dims(tf.expand_dims(term_1,-1),-1)
        shaper  = tf.constant([1,1,3,3])
        term_1  = tf.tile(term_1, shaper) # (?,nMaxwell,3,3)
        
        term_2 = tf.math.exp(-delta_t/(2.0*tau))
        # term_2 = g_bar*tau_bar/ delta_t *(1.0 - tf.math.exp(-delta_t/tau_bar)) # alternative update rule
        term_2 = tf.expand_dims(tf.expand_dims(term_2,-1),-1)
        shaper = tf.constant([1,1,3,3])
        term_2 = tf.tile(term_2, shaper) # (?,nMaxwell,3,3)
        
        delta_S_a = x1[2]-x0[2]
        # delta_S_a = tf.expand_dims(delta_S_a,1)
        # shaper = tf.constant([1,self.nMaxwell,1,1])
        # delta_S_a = tf.tile(delta_S_a, shaper) # (?,nMaxwell,3,3)
        
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
    

    ### comment in for pure relaxation test
    # Q_terms = g_i * tf.math.exp(-tf.reshape(time, [-1,1])/tau)
    # Q_sum = tf.math.reduce_sum(Q_terms, axis=2, name='sum_h_i', keepdims=True)
    # stress = g_infy + Q_sum
    
    return Q_ra



nMaxwell = 3

S_a = np.eye(3, dtype='float64')
S_a = np.expand_dims(np.expand_dims(np.expand_dims(S_a,0),0),0)
shaper = [batch_size,nSteps,nMaxwell,1,1]
S_a = np.tile(S_a, shaper)
S_a[:,0,:,:,:] = np.eye(3)
S_a = tf.constant(S_a)


tau = tf.constant([0.1,1,10], dtype='float64')
tau = tf.expand_dims(tf.expand_dims(tau,0),0,)
shaper = tf.constant([batch_size,nSteps,1])
tau = tf.tile(tau, shaper)

t = tf.expand_dims(tf.constant(np.linspace(0,1,nSteps)**2*10, dtype='float64'),0)
t = tf.tile(t,[batch_size,1])

inps = [S_a, t, tau]

#%%

Q = CM.PronySeries_Liu(inps, nMaxwell, nSteps)

q = Q[0,:,:,0,0].numpy()
t_plt = t[0].numpy()

plt.plot(time.reshape(-1), q[:,0])

#%%

S_neq = CM.compute_S_a_neq_bar(*inputs)

#%%
c1 = 6.0/3
c2 = 2.0/3

eye = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.eye(3,dtype=tf.float64), 0) , 0), 0)
eye = tf.tile(eye, (1,nSteps,nMaxwell,1,1))

trQ = tf.expand_dims(tf.expand_dims(tf.linalg.trace(Q),-1),-1)
trQ = tf.tile(trQ, (1,1,1,3,3))

t = tf.tile(tf.expand_dims(time,-1), (1, 1, nMaxwell))

decay  = tf.math.exp(-t/tau)
decay  = tf.expand_dims(tf.expand_dims(decay,-1),-1)
shaper  = tf.constant([1,1,1,3,3])
decay  = tf.tile(decay, shaper)



S = 2.0*(c1*eye*trQ - c2*Q)*decay


#%%

nMaxwell = 3


F = np.array([[1,0.1,0],[0,1,0],[0,0,1]])
F = np.expand_dims(np.expand_dims(F,0),0)
shaper = [batch_size,nSteps,1,1]
F = np.tile(F, shaper)
F = tf.Variable(F)


with tf.GradientTape(persistent=True) as tape:
    C = tf.linalg.matmul(tf.transpose(F,perm=(0,1,3,2)),F)
    I = tf.linalg.trace(C)
    dIdC = tape.gradient(I,C)

    # S_a = tf.expand_dims(tf.linalg.matmul(C,C),2)
    S_a = tf.expand_dims(C,2)
    shaper = tf.constant([1,1,nMaxwell,1,1])
    S_a = tf.tile(S_a, shaper)


    dS_a_dC = []
    for mm in range(nMaxwell):
        grads = []
        for ii in range(3):
            for jj in range(3):
                g = tape.gradient(S_a[:,:,mm,ii:ii+1,jj:jj+1], C) # (?,nSteps,3,3)
                # g = tf.divide(g, 2.0)
                grads.append(g)
                
        grads = tf.stack(grads, axis=2) # (?,nSteps,9,3,3)
        gradients = tf.reshape(grads, (batch_size, nSteps,3,3,3,3)) # (?,nSteps,3,3,3,3) #???
        gradients_t = 
        dS_a_dC.append(gradients)




#%%

for x,y in valDs:
    x,y

F = x[0][3:4]
t = x[1][3:4]

inps = [F,t]
res = model_full.predict(inps)

S_a = res[-17]
tau = res[17]
inps = [S_a, t, tau]
                     
Q = CM.stressUpdate_Liu(inps, nMaxwell, nSteps)

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)),Q[0,:,ii,0,0])

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)),S_a[0,:,ii,0,0])
    
    
dPsi_a_dJ_c    = res[-5]
ddPsi_a_ddI_c  = res[-4]
ddPsi_a_ddJ_c  = res[-3]
ddPsi_a_dIdJ_c = res[-2]
H              = res[-1]
C_bar          = np.matmul(np.transpose(F,(0,1,3,2)), F)
        
inp = [dPsi_a_dJ_c, ddPsi_a_ddI_c, ddPsi_a_ddJ_c, ddPsi_a_dIdJ_c, C_bar, Q, H[:,:,0,:,:]]
S_a_neq_bar_ex = CM.compute_S_a_neq_bar(*inp)             

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), S_a_neq_bar_ex[0,:,ii,0,0])
    
plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), tau[0,:,ii])

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), dPsi_a_dJ_c[0,:,ii])

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), ddPsi_a_ddJ_c[0,:,ii])

plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), ddPsi_a_ddI_c[0,:,ii])
    
plt.figure()
for ii in range(nMaxwell):
    plt.plot(tf.reshape(t,(-1)), ddPsi_a_dIdJ_c[0,:,ii])
