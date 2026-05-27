# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:37:09 2022

@author: Kian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
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
plt.close('all')

scale_temp = True

THETA = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0]
theta_max = np.max(THETA)
LAM = [2,3,4]
LAM_DOT = [0.03, 0.05, 0.1]

batch_size = 250

for theta in THETA:
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\\lambda$ [-]')
    ax.set_ylabel('Nominal stress $P$ [MPa]')
    for lam in LAM:
        colors = iter(prop_cycle.by_key()['color'])
        for ii, lam_dot in enumerate(LAM_DOT):
            c = next(colors)
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot)
            if not os.path.isfile(fname):
                continue
            
            data = np.loadtxt(fname, delimiter=',')
            
            stretch = data[:,0]
            stress = data[:,1]
            dlam = np.abs(np.diff(stretch))
            
            dt = dlam/lam_dot
            time = np.concatenate([np.array([0]),np.cumsum(dt)])
            
            f = interp1d(time, [stretch, stress])
            time_new = np.linspace(time[0], time[-1], batch_size)
            y_new = f(time_new)
            stretch_new, stress_new = y_new[0,:], y_new[1,:]   
            
            plt.plot(stretch_new, stress_new, color=c, linewidth=1, label='$\\dot{{\\lambda}}$ = {:} s$^{{-1}}$ '.format(lam_dot))
            plt.scatter(stretch, stress, color=c )#label='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$ - Data'.format(lam_dot))
            
            if scale_temp:
                theta_data = np.ones_like(stretch_new)*theta/theta_max 
            else:
                theta_data = np.ones_like(stretch_new)*theta
                
            d = np.c_[stretch_new, time_new, theta_data, stress_new]
            
            np.save('.\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot), d)
    
    ax.legend(title='{:.0f} °C'.format(theta))
    # plt.savefig('.\\temperature_{:}.pdf'.format(theta), format='pdf')
    
    plt.close()

#%% MAKE TRAINING DATA SET

THETA_TRAIN = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0] # [0.0, 10.0, 20.0, 40.0, 80.0]
LAM_TRAIN = [2, 4] # [3, 4] 
LAM_DOT_TRAIN = [0.03, 0.05, 0.1]

np.save('.\\data_split\\theta_train.npy', THETA_TRAIN)
np.save('.\\data_split\\lam_train.npy', LAM_TRAIN)
np.save('.\\data_split\\lam_dot_train.npy', LAM_DOT_TRAIN)

train_combinations = []

data_tr = []
for theta in THETA_TRAIN:
    for lam in LAM_TRAIN:
        for lam_dot in LAM_DOT_TRAIN:
            
            # ### for extrapolation with one curve at the temperature boundaries
            # if theta == 0.0 and ((lam == 2.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            # if theta == 80.0 and ((lam != 4.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue

            # ### for extrapolation without curves at the temperature boundaries
            # if theta == 0.0 or theta == 80.0:
            #     continue
        
            ### as in the paper's main part
            if lam_dot == 0.05:
                continue
            
            fname = '.\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot)
            if not os.path.isfile(fname):
                continue
            
            train_combinations.append([theta, lam, lam_dot])            
            
            d = np.load(fname)
            data_tr.append(d)    

data_tr = np.vstack(data_tr)
np.save('.\\trainData.npy', data_tr)
np.savetxt('.\\n_time_steps.txt', np.array([batch_size]))

train_combinations = np.vstack(train_combinations)

inps = (data_tr[:,0], data_tr[:,1], data_tr[:,2]) # stretch, time, temperature
outs = (data_tr[:,3])
ds_train = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
tf.data.experimental.save(ds_train, '.\ds_train', compression='GZIP')



#%% PLOT TRAINING DATA SET

# colors = ['tab:blue', 'tab:orange', 'tab:green']
captions = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

fig, axes = plt.subplots(3, 2, figsize=(14, 26))
axes = axes.flatten()

for ii, theta in enumerate(THETA_TRAIN):
    axes[ii].set_xlabel('Stretch $\\lambda$ [-]')
    axes[ii].set_ylabel('Nominal stress $P$ [MPa]')
    for lam in LAM_TRAIN:
        colors = iter(prop_cycle.by_key()['color'])
        for jj, lam_dot in enumerate(LAM_DOT_TRAIN):
            c = next(colors)
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot)
            
            
            # ### for extrapolation with one curve at the temperature boundaries
            # if theta == 0.0 and ((lam == 2.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue
            # if theta == 80.0 and ((lam != 4.0 or lam == 3.0) or (lam_dot == 0.03 or lam_dot == 0.05)):
            #     continue

            # ### for extrapolation without curves at the temperature boundaries
            # if theta == 0.0 or theta == 80.0:
            #     continue
            
            ### as in the paper's main part
            if lam_dot == 0.05:
                continue
            
            if not os.path.isfile(fname):
                continue
                
            data = np.loadtxt(fname, delimiter=',')
            stretch = data[:,0]
            stress = data[:,1]
               
            s = axes[ii].scatter(stretch, stress, s=15, color=c)
            if lam == 4:
                l = axes[ii].plot(stretch, stress, color=c, linewidth=1, label=format(lam_dot))          
            else:
                l = axes[ii].plot(stretch, stress, color=c, linewidth=1)

    axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
    axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ [s$^{-1}$]')
    axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))

plt.tight_layout()
plt.savefig('.\\training_data.pdf', format='pdf')    



#%% VALIDATION DATA

THETA_VAL = [0.0, 10.0, 20.0, 40.0, 60.0, 80.0]
LAM_VAL = [3, 4] # [2, 4]
LAM_DOT_VAL = [0.03, 0.05, 0.1]

np.save('.\\data_split\\theta_val.npy', THETA_VAL)
np.save('.\\data_split\\lam_val.npy', LAM_VAL)
np.save('.\\data_split\\lam_dot_val.npy', LAM_DOT_VAL)

valid_combinations = []

data_val = []

for theta in THETA_VAL:
    for lam in LAM_VAL:
        for lam_dot in LAM_DOT_VAL:
            
            ### for extrapolation with one curve at the temperature boundaries
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue
            # if lam_dot == 0.1 and lam == 4.0:
            #     continue

            # ### for extrapolation without curves at the temperature boundaries
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue
            
            ### as in the paper's main part
            if lam_dot == 0.03:
                continue
            if lam == 4 and lam_dot !=0.05:
                continue
                
            fname = '.\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot)
            if not os.path.isfile(fname):
                continue
            
            valid_combinations.append([theta, lam, lam_dot])
            
            d = np.load(fname)
            data_val.append(d)    

data_val = np.vstack(data_val)
np.save('.\\trainData.npy', data_val)
np.savetxt('.\\n_time_steps.txt', np.array([batch_size]))

valid_combinations = np.vstack(valid_combinations)

inps = (data_val[:,0], data_val[:,1], data_val[:,2]) # stretch, time, normalized temperature
outs = (data_val[:,3])
ds_valid = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
tf.data.experimental.save(ds_valid, '.\ds_valid', compression='GZIP')


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
            
            fname = '.\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot)
            if not os.path.isfile(fname):
                continue 

            ### for extrapolation with one curve at the temperature boundaries
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue
            # if lam_dot == 0.1 and lam == 4.0:
            #     continue
        
            # ### for extrapolation without curves at the temperature boundaries
            # if not ((theta != 0.0) ^ (theta != 80.0)):
            #     continue

            ### as in the paper's main part
            if lam_dot == 0.03:
                continue
            if lam == 4 and lam_dot !=0.05:
                continue
                            
           
            data = np.loadtxt(fname, delimiter=',')
            stretch = data[:,0]
            stress = data[:,1]
                
            axes[ii].scatter(stretch, stress, s=15, color=c)
            axes[ii].plot(stretch, stress, color=c, linewidth=1, label=format(lam_dot))          

    axes[ii].text(.5, -.18, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
    axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ [s$^{-1}$]')
    axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))

plt.tight_layout()
plt.savefig('.\\validation_data.pdf', format='pdf')    


#%%

plt.rc('legend', fontsize=16)    # fontsize of the tick labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('axes', labelsize=30)   

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('Stretch $\lambda$')
ax.set_ylabel('Stretch rate $\dot{\lambda}$')

ax.xaxis.labelpad=10
ax.yaxis.labelpad=15
ax.zaxis.labelpad=10

ax.xaxis.set_ticks(np.arange(2, 5, 1))
# ax.xaxis.set_ticks(np.arange(0, 100, 20))
# ax.xaxis.set_ticks(np.arange(0, 100, 20))
#ax.set_ylim([0.03,0.1])
#ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False)

# Get rid of the panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.scatter(train_combinations[:,1], train_combinations[:,2], train_combinations[:,0], s=25, color='tab:blue', depthshade=False, label='Training data')
ax.scatter(valid_combinations[:,1], valid_combinations[:,2], valid_combinations[:,0], s=25, color='tab:orange',  depthshade=False, label='Validation data')

ax.legend(loc='lower center', bbox_to_anchor=(0.11, 0.8))
ax.view_init(13, -145, 0) 
ax.zaxis.set_rotate_label(False) 
ax.set_zlabel('Temperature $\Theta$', rotation=90)

plt.title('Interpolation')
plt.savefig('.\\interpolation_combinations.pdf', format='pdf')    


# import plotly.io as pio
# import plotly.express as px
# pio.renderers.default='browser'

# import plotly.graph_objects as go

# convex_hull = ConvexHull(train_combinations)
# xc = train_combinations[convex_hull.vertices]

# fig = go.Figure()
# fig.add_trace(go.Mesh3d(x=xc[:, 0], 
#                         y=xc[:, 1], 
#                         z=xc[:, 2], 
#                         color="blue", 
#                         opacity=1,
#                         alphahull=0))

# fig.scatter()


# fig.show()




