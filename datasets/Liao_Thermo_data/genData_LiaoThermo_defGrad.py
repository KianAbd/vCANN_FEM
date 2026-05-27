# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:37:09 2022

@author: Kian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_smoothing_spline
import tensorflow as tf
import os

#%%

SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

prop_cycle = plt.rcParams['axes.prop_cycle']
linestyles = ['-', '-.', '--']


#%%


def defGrad(lam): # Deformation gradient for incompressible uniaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]    
    F = np.zeros([nSamples, nSteps, 3, 3])
    F[:,:,0,0] = lam
    F[:,:,1,1] = 1.0/(np.sqrt(lam))
    F[:,:,2,2] = 1.0/(np.sqrt(lam))
    
    return F

#

def defGrad_dot(lam, lam_dot): # time derivative of the deformation gradient for incompressible uniaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]    
    F_dot = np.zeros([nSamples, nSteps, 3, 3])
    F_dot[:,:,0,0] = lam_dot
    F_dot[:,:,1,1] = -lam_dot/(2.*lam**(3./2.))
    F_dot[:,:,2,2] = -lam_dot/(2.*lam**(3./2.))
    
    return F_dot

#%%
# plt.close('all')

scale_temp = False
rateDependent = True
with_feature = True

res_1 = 1.2 # scaling exponent; controls non-uniform spacing at the beginning of loading and unloading
res_2 = 1.2

THETA = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0]
theta_max = np.max(THETA)
LAM = [2.0, 3.0, 4.0]
LAM_DOT = [0.03, 0.05, 0.1]

nSteps = 800
np.savetxt('.\\n_time_steps.txt', [nSteps])

for theta in THETA:
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\\lambda$ [-]')
    ax.set_ylabel('Nominal stress $P$ [kPa]')
    
    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Time $t$ [-]')
    ax.set_ylabel('Stretch $\\lambda$ [-]')
    
    for lam in LAM:
        colors = iter(prop_cycle.by_key()['color'])
        for ii, lam_dot in enumerate(LAM_DOT):
            c = next(colors)
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot)
            if not os.path.isfile(fname):
                continue
            
            data = np.loadtxt(fname, delimiter=',')
            
            stretch = data[:,0]
            stress = data[:,1] * 100

            arg_stretch_max = np.argmax(stretch)
            arg_stress_max = np.argmax(stress)
            if arg_stress_max != arg_stretch_max:
                raise ValueError("'arg_stretch_max' and 'arg_stretch_max' do not coincide."
                                )
            dlam = np.abs(np.diff(stretch))
            dt = dlam/lam_dot
            interps = [stretch, stress]

            stretch_rate = np.full(stretch.shape, lam_dot)
            stretch_rate[arg_stretch_max+1:] = -lam_dot
            stretch_rate[0] = 1.e-7 # to avoid zero division
            interps.append(stretch_rate)
        
            time = np.concatenate([np.array([0]),np.cumsum(dt)])
            f = interp1d(time, interps)
            # f_lam = make_smoothing_spline(time, stretch)
            # f_str = make_smoothing_spline(time, stress)
            # f_lam_dot = interp1d(time, stretch_rate)

            # higher resolution for initial loading phase and load reversal
            t_load = np.linspace(time[0], 1, nSteps//2)**res_1 * time[arg_stress_max]
            t_unload = np.linspace(0, 1, nSteps//2+1)**res_2 * (time[-1]-time[arg_stress_max]) + time[arg_stress_max]
            t_unload = t_unload[1:]
            time_new = np.concatenate([t_load, t_unload])
            
            y_new = f(time_new)
            # stretch_new = f_lam(time_new)
            # stress_new = f_str(time_new)
            
            stretch_new, stress_new = y_new[0,:], y_new[1,:]
            
            if rateDependent:
                # stretch_rate_new = f_lam_dot(time_new)
                stretch_rate_new = np.ones_like(time_new)*lam_dot
                stretch_rate_new[len(t_load):] = -lam_dot
                stretch_rate_new[0] = 1.e-7
            
            ax.plot(stretch_new, stress_new, color=c, linewidth=1, label='$\\dot{{\\lambda}}$ = {:} s$^{{-1}}$ '.format(lam_dot))
            ax.scatter(stretch, stress, color=c )#label='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$ - Data'.format(lam_dot))
            
            ax1.scatter(time_new, stretch_rate_new, color=c )#label='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$ - Data'.format(lam_dot))
            ax1.scatter(time, stretch_rate, color=c )#label='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$ - Data'.format(lam_dot))

            if scale_temp:
                theta_data = -np.ones_like(stretch_new)*theta/theta_max 
            else:
                theta_data = -np.ones_like(stretch_new)*theta
                
            d = np.c_[stretch_new, time_new, theta_data, stress_new]
            if rateDependent:
                d = np.concatenate([d, stretch_rate_new[:, np.newaxis]], axis=-1)        
            
            np.save('.\\{:}\\{:}_{:}.npy'.format(theta, int(lam), lam_dot), d)
    
    ax.legend(title='{:.0f} °C'.format(theta))
    # fig.savefig('.\\temperature_{:}.pdf'.format(theta), format='pdf')
    
    # plt.close()

#%% MAKE TRAINING DATA SET

THETA_TRAIN = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0] # [0.0, 10.0, 20.0, 40.0, 80.0]
LAM_TRAIN = [2.0, 4.0,] # [2, 3, 4] # [3, 4] 
LAM_DOT_TRAIN = [0.03, 0.1] # [0.03, 0.05, 0.1]

np.save('.\\data_split\\theta_train.npy', THETA_TRAIN)
np.save('.\\data_split\\lam_train.npy', LAM_TRAIN)
np.save('.\\data_split\\lam_dot_train.npy', LAM_DOT_TRAIN)

data_tr = []
for theta in THETA_TRAIN:
    for lam in LAM_TRAIN:
        for lam_dot in LAM_DOT_TRAIN:
            
            # ### for extrapolation
            # if theta == 0.0 and ((lam == 2.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            # if theta == 80.0 and ((lam != 4.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            
            ### as in the paper's main part
            if lam_dot == 0.05:
                continue
            
            # if lam_dot == 0.1 and lam == 2:
            #     continue
            
            fname = '.\\{:}\\{:}_{:}.npy'.format(theta, int(lam), lam_dot)
            if not os.path.isfile(fname):
                continue
            
            d = np.load(fname)
            data_tr.append(d)    

data_tr = np.vstack(data_tr)

# make dataset with deformation gradient (3D)
lam_reshape = data_tr[:,0].reshape(-1,nSteps) # (?, nSteps)
t_reshape = data_tr[:,1].reshape(-1,nSteps) # (?, nSteps)
theta = np.expand_dims(data_tr[:,2].reshape(-1,nSteps), axis=-1) # (?, nSteps, 1)

P11_reshape = data_tr[:,3].reshape(-1,nSteps) # (?, nSteps)
P = np.zeros([P11_reshape.shape[0],nSteps,3,3])
P[:,:,0,0] = P11_reshape

F_tr = defGrad(lam_reshape) # (?, nSteps, 3, 3)
if rateDependent:
    F_dot_tr = defGrad_dot(lam_reshape, data_tr[:,4].reshape(-1,nSteps)) # (?, nSteps, 3, 3)
    inps = [F_tr, t_reshape, F_dot_tr,]
else:
    inps = [F_tr, t_reshape,]
    
if with_feature:
    inps.append(theta)

inps = tuple(inps)
    
outs = (P)

batchSize = len(data_tr)//nSteps
ds_train = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batchSize)
tf.data.experimental.save(ds_train, '.\ds_train_defGrad', compression='GZIP')



#%% PLOT TRAINING DATA SET

# colors = ['tab:blue', 'tab:orange', 'tab:green']
captions = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

fig, axes = plt.subplots(3, 2, figsize=(14, 26))
axes = axes.flatten()

for ii, theta in enumerate(THETA_TRAIN):
    axes[ii].set_xlabel('Stretch $\\lambda$ [-]')
    axes[ii].set_ylabel('Nominal stress $P$ [kPa]')
    for lam in LAM_TRAIN:
        colors = iter(prop_cycle.by_key()['color'])
        for jj, lam_dot in enumerate(LAM_DOT_TRAIN):
            c = next(colors)
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot)
            
            
            # ### for extrapolation
            # if theta == 0.0 and ((lam == 2.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            # if theta == 80.0 and ((lam != 4.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            
            ### as in the paper's main part
            if lam_dot == 0.05:
                continue
            
            # if lam_dot == 0.1 and lam == 2:
            #     continue
            
            if not os.path.isfile(fname):
                continue
                
            data = np.loadtxt(fname, delimiter=',')
            stretch = data[:,0]
            stress = data[:,1]
               
            if lam == 4:
                axes[ii].plot(stretch, stress, color=c, marker='.', markersize=15, linewidth=1, label=format(lam_dot))          
            else:
                axes[ii].plot(stretch, stress, color=c, marker='.', markersize=15, linewidth=1)
                
            # d = np.load('.\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot))
            # stretch, stress = d[:,0], d[:,3]
            # axes[ii].plot(stretch, stress, color=c, marker='.', markersize=5, linewidth=1)
            
            
    axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
    axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ [s$^{-1}$]')
    axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))

plt.tight_layout()
plt.savefig('.\\training_data.pdf', format='pdf')    



#%% VALIDATION DATA

THETA_VAL = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0]
LAM_VAL = [3.0, 4.0] # [2,3,4] # [2, 4]
LAM_DOT_VAL = [0.03, 0.05, 0.1]

np.save('.\\data_split\\theta_val.npy', THETA_VAL)
np.save('.\\data_split\\lam_val.npy', LAM_VAL)
np.save('.\\data_split\\lam_dot_val.npy', LAM_DOT_VAL)


data_val = []

for theta in THETA_VAL:
    for lam in LAM_VAL:
        for lam_dot in LAM_DOT_VAL:
            
            # ### for extrapolation
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue
            # if lam_dot == 0.1 and lam == 4.0:
            #     continue
            
            # ## as in the paper's main part
            if lam_dot == 0.03:
                continue
            if lam == 4 and lam_dot !=0.05:
                continue
                
            fname = '.\\{:}\\{:}_{:}.npy'.format(theta, int(lam), lam_dot)
            if not os.path.isfile(fname):
                continue
            
            d = np.load(fname)
            data_val.append(d)    

data_val = np.vstack(data_val)

# make dataset with deformation gradient (3D)
lam_reshape = data_val[:,0].reshape(-1,nSteps) # (?, nSteps)
t_reshape = data_val[:,1].reshape(-1,nSteps) # (?, nSteps)
theta = np.expand_dims(data_val[:,2].reshape(-1,nSteps), axis=-1) # (?, nSteps, 1)

P11_reshape = data_val[:,3].reshape(-1,nSteps) # (?, nSteps)
P = np.zeros([P11_reshape.shape[0],nSteps,3,3])
P[:,:,0,0] = P11_reshape

F_val = defGrad(lam_reshape) # (?, nSteps, 3, 3)
if rateDependent:
    F_dot_val = defGrad_dot(lam_reshape, data_val[:,4].reshape(-1,nSteps)) # (?, nSteps, 3, 3)
    inps = [F_val, t_reshape, F_dot_val]
else:
    inps = [F_val, t_reshape]
    
if with_feature:
    inps.append(theta)
    
inps = tuple(inps)

outs = (P)

batchSize = len(data_val)//nSteps
ds_val = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batchSize)
tf.data.experimental.save(ds_val, '.\ds_valid_defGrad', compression='GZIP')


#%% PLOT VALIDATION DATA SET

fig, axes = plt.subplots(3, 2, figsize=(14, 26))
axes = axes.flatten()

for ii, theta in enumerate(THETA_VAL):
    axes[ii].set_xlabel('Stretch $\\lambda$ [-]')
    axes[ii].set_ylabel('Nominal stress $P$ [MPa]')
    for lam in LAM_VAL:
        colors = iter(prop_cycle.by_key()['color'])
        for jj, lam_dot in enumerate(LAM_DOT_VAL):
            c = next(colors)
            
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot)
            if not os.path.isfile(fname):
                continue 

            # ### for extrapolation
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue
            # if lam_dot == 0.1 and lam == 4.0:
            #     continue

            ### as in the paper's main part
            if lam_dot == 0.03:
                continue
            if lam == 4 and lam_dot !=0.05:
                continue
                            
           
            data = np.loadtxt(fname, delimiter=',')
            stretch = data[:,0]
            stress = data[:,1]
                
            axes[ii].plot(stretch, stress, color=c, marker='.', markersize=15, linewidth=1, label=format(lam_dot))          

    axes[ii].text(.5, -.18, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
    axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ [s$^{-1}$]')
    axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))

plt.tight_layout()
plt.savefig('.\\validation_data.pdf', format='pdf')    










