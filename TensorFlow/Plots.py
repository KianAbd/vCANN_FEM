# Standard imports

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.axis3d import Axis
from plotly.subplots import make_subplots
from natsort import natsorted
from utils import R2


import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.ticker as ticker
import json
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

#%%


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('text', usetex=True)

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['text.latex.preamble'] = r'\usepackage[cm]{sfmath}'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'cm'

prop_cycle = plt.rcParams['axes.prop_cycle']


linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 5)),
     'densely dotted':        (0, (1, 1)),

     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}


### Patch to remove margins from 3D plots
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
### Patch end

#%%


def defGrad(lam): # Deformation gradient for incompressible uniaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]    
    F = np.zeros([nSamples, nSteps, 3, 3])
    F[:,:,0,0] = lam
    F[:,:,1,1] = 1.0/(np.sqrt(lam))
    F[:,:,2,2] = 1.0/(np.sqrt(lam))
    
    return F

def defGrad_dot(lam, lam_dot): # time derivative of the deformation gradient for incompressible uniaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]
    F_dot = np.zeros([nSamples, nSteps, 3, 3])
    F_dot[:,:,0,0] = lam_dot
    F_dot[:,:,1,1] = -lam_dot/(2.*lam**(3./2.))
    F_dot[:,:,2,2] = -lam_dot/(2.*lam**(3./2.))
    
    return F_dot


def defGrad_bi(lam): # Deformation gradient for incompressible biaxial tension loading [?,3,3]
    nSamples = lam.shape[0]
    nSteps = lam.shape[1]    
    F = np.zeros([nSamples, nSteps, 3, 3])
    F[:,:,0,0] = lam
    F[:,:,1,1] = lam
    F[:,:,2,2] = 1.0/lam**2
    
    return F
    
    
def defGrad_ss(gamma): # Deformation gradient for incompressible simple shear loading [?,3,3]
    nSamples = gamma.shape[0]
    nSteps = gamma.shape[1]    
    F = np.zeros([nSamples, nSteps, 3, 3])
    F[:,:,0,0] = 1.0
    F[:,:,1,1] = 1.0
    F[:,:,2,2] = 1.0
    F[:,:,0,1] = gamma
    
    return F

#%%
def plot_multi_step(model, ds, outputFolder):
    
    titles = ['Equi-biaxial', 'Strip-x', 'Strip-y', 'Off-x', 'Off-y']
    tick_spacing = 500

    fig = plt.figure(figsize=(14.9,13.6))
    gs = fig.add_gridspec(4, 6, height_ratios=[2,1,2,1], hspace=0.17, wspace=0.4)
    
    # Create axes for stress plots
    ax_stress = [
        fig.add_subplot(gs[0, 0:2]),  # Equi-biaxial
        fig.add_subplot(gs[0, 2:4]),  # Strip-x
        fig.add_subplot(gs[0, 4:6]),  # Strip-y
        fig.add_subplot(gs[2, 1:3]),  # Off-x
        fig.add_subplot(gs[2, 3:5])   # Off-y
    ]
    
    # Create axes for stretch history
    ax_stretch = [
        fig.add_subplot(gs[1, 0:2]),  # Equi-biaxial stretch
        fig.add_subplot(gs[1, 2:4]),  # Strip-x stretch
        fig.add_subplot(gs[1, 4:6]),  # Strip-y stretch
        fig.add_subplot(gs[3, 1:3]),  # Off-x stretch
        fig.add_subplot(gs[3, 3:5])   # Off-y stretch
    ]
    
    R2_ = []
    MSE = []
    
    for x,y in ds:

        stress = model.predict(x)
        stress_data = y.numpy()
        numBatches = x[0].get_shape().as_list()[0]

        stress_err = np.hstack([stress[:,:,0,0], stress[:,:,1,1]])
        stress_data_err = np.hstack([stress_data[:,:,0,0], stress_data[:,:,1,1]])

        for kk in range(numBatches):
            r2 = R2(stress_data_err[kk], stress_err[kk])
            R2_.append(r2)
            mse = ((stress_err[kk] - stress_data_err[kk])**2).mean(axis=None) 
            MSE.append(mse)
            time = x[1][kk]

            n=10
            s=25
            ax_stress[kk].plot(time, stress[kk,:,0,0], '-', color='tab:red', label=r"$P_{11}$")       
            ax_stress[kk].plot(time, stress[kk,:,1,1], '-', color='tab:blue', label=r"$P_{22}$")   
            ax_stress[kk].scatter(time[::n], stress_data[kk,::n,0,0], s=s, facecolors='none', color='tab:red', label=r"$\hat{P}_{11}$")       
            ax_stress[kk].scatter(time[::n], stress_data[kk,::n,1,1], s=s, facecolors='none', color='tab:blue', label=r"$\hat{P}_{22}$") 
            
            if kk == 0 or kk == 3:
                ax_stress[kk].set_ylabel(r'Nominal Stress $P_{ij}$ [MPa]')
            t = titles[kk] + '\n$R^2$={:5.3f}, MSE={:.3E}'.format(r2, mse)
            ax_stress[kk].set_title(t)
            ax_stress[kk].grid(True)
            
            # stretch history
            if titles[kk] == 'Off-y' or titles[kk] == 'Strip-y' :
                lam = x[0][kk,:,1,1]
            else:
                lam = x[0][kk,:,0,0]
            ax_stretch[kk].set_xlabel(r'Time $t$ [s]')
            if kk == 0 or kk == 3:
                ax_stretch[kk].set_ylabel(r'Stretch $\lambda$ [-]')            
            ax_stretch[kk].grid(True)
            ax_stretch[kk].plot(time, lam, '-', color='tab:grey')
            ax_stretch[kk].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    # Add shared x-axis labels for top row stress plots
    for ax in ax_stress:
        ax.set_xticklabels([])
    
    # Adjust spacing between rows
    plt.subplots_adjust(bottom=0.21)
    
    # Manually adjust spacing between second and third row
    # gs.update(hspace=0.0)
    for i in [0, 1]:  # First two rows
        pos = ax_stretch[i].get_position()
        ax_stretch[i].set_position([pos.x0, pos.y0, pos.width, pos.height])
    
    # Add extra space before third row
    offset = 0.1
    for i in [3, 4]:  # Bottom row axes
        pos = ax_stress[i].get_position()
        ax_stress[i].set_position([pos.x0, pos.y0 - offset, pos.width, pos.height])
        pos = ax_stretch[i].get_position()
        ax_stretch[i].set_position([pos.x0, pos.y0 - offset, pos.width, pos.height])
    
    lines_labels = [ax_stress[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=4)   

    fig.savefig(outputFolder+'\\multi_step_general_biaxial.pdf', format='pdf', bbox_inches="tight")   
    # plt.show() 


def plot_multi_step_isotropic(model, ds, outputFolder):
    
    titles = ['Uniaxial', 'Equi-biaxial', 'Pure shear']
    tick_spacing = 500
    # fig, ax = plt.subplots(2, 3, figsize=(10,7.3), sharex='col', height_ratios=[2,1]) # sharey='row'
    fig, ax = plt.subplots(2, 3, figsize=(13,6.7), sharex='col', height_ratios=[2,1]) # sharey='row'
    ax = ax.flatten()
    
    R2_ = []
    MSE = []
    
    for x,y in ds:

        stress = model.predict(x)
        stress_data = y.numpy()
        numBatches = x[0].get_shape().as_list()[0]

        stress_err = np.hstack([stress[:,:,0,0], stress[:,:,1,1]])
        stress_data_err = np.hstack([stress_data[:,:,0,0], stress_data[:,:,1,1]])

        for kk in range(3):
            r2 = R2(stress_data_err[kk], stress_err[kk])
            R2_.append(r2)
            mse = ((stress_err[kk] - stress_data_err[kk])**2).mean(axis=None) 
            MSE.append(mse)
            time = x[1][kk]

            n=10
            s=25
            ax[kk].plot(time, stress[kk,:,0,0], '-', color='tab:red', label=r"$P_{11}$")      
            if titles[kk] == 'Pure shear':
                ax[kk].plot(time, stress[kk,:,1,1], '-', color='tab:blue', label=r"$P_{22}$")   
            ax[kk].scatter(time[::n], stress_data[kk,::n,0,0], s=s, facecolors='none', color='tab:red', label=r"$\hat{P}_{11}$")  
            if titles[kk] == 'Pure shear':
                ax[kk].scatter(time[::n], stress_data[kk,::n,1,1], s=s, facecolors='none', color='tab:blue', label=r"$\hat{P}_{22}$") 
            

            if kk == 0:
                ax[kk].set_ylabel(r'Nominal Stress $P_{ij}$ [MPa]')
            #ax[idx].legend()
            t = titles[kk] + '\n$R^2$={:5.3f}, MSE={:.3E}'.format(r2, mse)
            ax[kk].set_title(t)
            ax[kk].grid(True)
            
            # stretch history
            lam = x[0][kk,:,0,0]
            ax[kk+3].set_xlabel(r'Time $t$ [s]')
            if kk == 0:
                ax[kk+3].set_ylabel(r'Stretch $\lambda$ [-]')            
            ax[kk+3].grid(True)
            ax[kk+3].plot(time, lam, '-', color='tab:grey')      

            ax[kk+3].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            # ax[kk+3].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    
    lines_labels = [ax[2].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='lower center', ncol=4)  
    
    # ax[0].axis('off')
    # ax[4].axis('off')
    # ax[5].axis('off')
    # ax[9].axis('off')

    fig.savefig(outputFolder+'\\multi_step_isotropic.pdf', format='pdf', bbox_inches="tight") 


def plot_struc_tensor(model_fit, extra_struc=None, plotly=False, outputFolder=None):

    inputs=model_fit.get_layer('F_input').input
    outputs=model_fit.get_layer('H').output
    model_H = tf.keras.models.Model(inputs, outputs)

    numTens = model_H.get_layer('H').input[-1].numpy()
    nSteps = model_fit.outputs[0].shape.as_list()[1]
    numDir = model_H.get_layer('H').input[-2].numpy()

    nTheta, nPhi = 100, 100

    theta, phi = np.linspace(0, np.pi, nTheta), np.linspace(0, 2*np.pi, nPhi)
    THETA, PHI = np.meshgrid(theta, phi)

    X = np.sin(THETA)*np.cos(PHI)
    Y = np.sin(THETA)*np.sin(PHI)
    Z = np.cos(THETA)

    P = np.array([X,Y,Z])
    P = P.reshape(3,-1)

    if extra_struc != 0:
        F = np.eye(3) # dummy deformation gradient 
        F = np.expand_dims(F,(0,1))
        F = np.tile(F, (1,nSteps,1,1))    
        Hs = model_H(F)        
    else:
        extra_struc = np.expand_dims(extra_struc, axis=(0,1))
        extra_struc = np.tile(extra_struc, (1,nSteps,1))
        Hs = model_H(extra_struc)

    Hs = Hs[0,0]
    Hs = np.split(Hs, numTens, 0)

    x_max = 1.
    y_max = 1.
    z_max = 1.
    abs_H_max = 0.0
    
    for ii in range(numTens):
        H = Hs[ii].squeeze()
        abs_H = np.einsum('im,ik,km->m',P, H, P)
        abs_H = abs_H.reshape(nPhi, nTheta)
        abs_H_max = np.maximum(abs_H_max, np.max(abs_H))
        
        X_ = abs_H*np.sin(THETA)*np.cos(PHI)
        Y_ = abs_H*np.sin(THETA)*np.sin(PHI)
        Z_ = abs_H*np.cos(THETA)

    fig, axes = plt.subplots(1,numTens,figsize=(24,8), subplot_kw=dict(projection='3d'))
    if numTens == 1:
        axes = [axes,]
    
    for ii in range(numTens):
        H = Hs[ii].squeeze()
        abs_H = np.einsum('im,ik,km->m',P, H, P)
        abs_H = abs_H.reshape(nPhi, nTheta)
        
        X_ = abs_H*np.sin(THETA)*np.cos(PHI)
        Y_ = abs_H*np.sin(THETA)*np.sin(PHI)
        Z_ = abs_H*np.cos(THETA)
                
        d = np.abs(abs_H/abs_H_max)


        ax = axes[ii]
        tick_spacing = 1.
        fontsize = 32
        labelpad=20
        
        ax.set_xlabel("x'", labelpad=labelpad, fontsize=fontsize)
        ax.set_ylabel("y'", labelpad=labelpad, fontsize=fontsize)
        ax.set_zlabel("z'", labelpad=labelpad, fontsize=fontsize)
        
        ax.tick_params(axis='both', which='major', labelsize=fontsize)    
        
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([-y_max, y_max])
        ax.set_zlim([-z_max, z_max])
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        cmap = cm.coolwarm(d)               
        ax.plot_surface(X_, Y_, Z_, facecolors=cmap, rcount=200, ccount=200, linewidth=0, antialiased=True)
        ax.set_aspect('equal')
        
        
        zdirs = ['x','y', 'z']
        for d in zdirs:
            if d == 'y':
                offset = 1
                # offset = ax.get_ylim()[1]
            elif d == 'x':
                offset = -1
                # offset = ax.get_xlim()[0]
            elif d == 'z':        
                offset = -1   
                # offset = ax.get_zlim()[0]
            ax.contour(X_, Y_, Z_, levels=[0,], zdir=d, offset=offset, colors='tab:grey')

        # ax.set_title('Generalized structural tensor $\\tilde{{\\mathbf{{L}}}}_{:}$'.format(ii+1))
    
    plt.tight_layout()
    if outputFolder:
        fig.savefig(outputFolder+'\\generalized_structural_tensor.pdf', format='pdf',
                    bbox_inches="tight",
                    #pad_inches = 0       # remove white space
                    )    

        
    if plotly:
        fig.layout['scene'] =  dict(
                xaxis = dict(title_text='x', range=[-x_max, x_max]),
                yaxis = dict(title_text='y', range=[-y_max, y_max]),
                zaxis = dict(title_text='z', range=[-z_max, z_max]),
                aspectmode = 'data'
                )
        
        for ii in range(2,numTens+1):
            fig.layout['scene{:}'.format(ii)] = dict(
                    xaxis = dict(title_text='x', range=[-x_max, x_max]),
                    yaxis = dict(title_text='y', range=[-y_max, y_max]),
                    zaxis = dict(title_text='z', range=[-z_max, z_max]),
                    aspectmode='data'
                    )
    
        fig.update_traces(showscale=False)
        fig.update_layout(height=600, width=2400, title_text="Generalized structural tensors")        
        fig.show()
                

        
    

def plot_elastic(model, trainDs, outputFolder, aniso=False, valDs=None, title='Total'):  
      
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    axes = ax.flatten()
    
    Ds = [trainDs, valDs]
    Title = ['Training', 'Validation']
    r2_avg = []
    
    for ds, ax, l in zip(Ds, axes, Title): 
        R2_ = []
        for x,y in ds:           

            stretch_true = x[0][:,:,0,0].numpy().squeeze()
            F = x[0]
            stress_true =  y.numpy().squeeze()
            stress_pred = model.predict(F)[:,:,0,0]
            
            for ii in range(len(stress_pred)):
                idxs = np.argsort(stretch_true[ii])
                stretch_sort = stretch_true[ii][idxs]
                stress_pred_sort = stress_pred[ii][idxs]
                stress_true_sort = stress_true[ii][idxs]
                
                # interpolate for equidistant scatter plots
                stretch_scat = np.linspace(stretch_sort[0], stretch_sort[-1], 50)
                f_scat = scipy.interpolate.interp1d(stretch_sort, stress_true_sort)
                stress_scat = f_scat(stretch_scat)
                
                ax.plot(stretch_sort, stress_pred_sort, label='Isotropic')
                    
                ax.scatter(stretch_scat, stress_scat, s=15)
        
                r2 = R2(stress_true[ii], stress_pred[ii])
                R2_.append(r2)
            
        r2_avg.append(np.mean(R2_))
        
        ax.set_ylabel(r'Nominal stress $P$ [MPa]')
        ax.set_xlabel(r'Stretch $\lambda$ [-]')
        ax.set_title(l, fontsize=MEDIUM_SIZE)

        if aniso == True:
            ax.legend(#bbox_to_anchor=(1.04,0.5), 
                       title='Fiber angle $\\varphi$ [°]',
                       #loc="center left", 
                       fancybox=True,
                       framealpha=0.8)
        else:
            ax.legend(#bbox_to_anchor=(1.04,0.5), 
                       #loc="center left", 
                       fancybox=True,
                       framealpha=0.8)

    
    text = r'$R^2_{{tr}}={:5.4f}$'.format(r2_avg[0])
    axes[0].text(0.5, 0.9,
                 text, 
                 horizontalalignment='center', 
                 verticalalignment='top',
                 color='k',
                 transform=axes[0].transAxes,
                 bbox=dict(facecolor='none', edgecolor='none'),
                 fontsize=SMALL_SIZE
             )
    
    text = r'$R^2_{{val}}={:5.4f}$'.format(r2_avg[1])
    axes[1].text(0.5, 0.9,
                 text, 
                 horizontalalignment='center', 
                 verticalalignment='top',
                 color='k',
                 transform=axes[1].transAxes,
                 bbox=dict(facecolor='none', edgecolor='none'),
                 fontsize=SMALL_SIZE
             )
    
    plt.suptitle('{:} elastic stress response'.format(title))
    plt.savefig(outputFolder+'\\{:}_elastic_stress_response.pdf'.format(title.lower()), format='pdf', bbox_inches="tight")

    
#
###


def plot_elastic_panel(model_stressElast, model_stressElastIso, model_stressElastAniso, model_Psi_iso, model_Psi_aniso,
                       validDs_elastic, validDs_elasticIso, validDs_elasticAniso, ds_psi_iso, ds_psi_aniso, outputFolder, aniso=False):  
          

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    captions = ['(a)','(b)','(c)','(d)']    
    title = ['Strain energy funciton', 'Total elastic stress', 'Isotropic elastic stress', 'Anisotropic elastic stress']
    
    def plot(model, ax, ds, aniso=False):
        for x,y in ds:           
            stretch_true = x[0].numpy().squeeze()   
            stress_true =  y.numpy().squeeze()
    
            stress_pred = model.predict(x, batch_size=len(stretch_true))
            stress_pred = stress_pred.reshape(-1)
            
            idxs = np.argsort(stretch_true)
            stretch_sort = stretch_true[idxs]
            stress_pred_sort = stress_pred[idxs]
            stress_true_sort = stress_true[idxs]
            
            n=25
            # interpolate for equidistant scatter plots
            stretch_scat = np.linspace(stretch_sort[0], stretch_sort[-1], n)
            f_scat = scipy.interpolate.interp1d(stretch_sort, stress_true_sort)
            stress_scat = f_scat(stretch_scat)
            
            if aniso == True:
                phi = x[1][0,0].numpy()/np.pi*180.
                ax.plot(stretch_sort, stress_pred_sort, lw=1, label='{:}'.format(phi))
            else:
                ax.plot(stretch_sort, stress_pred_sort, lw=1)
                
            ax.scatter(stretch_scat, stress_scat, s=20)
        
                    
    plot(model_Psi_iso,          axes[0], ds_psi_iso,           aniso=False)
    plot(model_Psi_aniso,        axes[0], ds_psi_aniso,         aniso=True)    
    plot(model_stressElast,      axes[1], validDs_elastic,      aniso=True)
    plot(model_stressElastIso,   axes[2], validDs_elasticIso,   aniso=False)
    plot(model_stressElastAniso, axes[3], validDs_elasticAniso, aniso=True)
    
        
    
    for ii, ax in enumerate(axes):
        ax.text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
        ax.set_title(title[ii], fontsize=MEDIUM_SIZE)
        if ii != 2:
            ax.legend(title=u'Fiber angle $\\varphi$ [°]')
        if ii ==0:
            ax.set_xlabel('Stretch $\\lambda$ [-]')
            ax.set_ylabel('Strain energy function $\Psi$ [MPa]')
        else:
            ax.set_xlabel('Stretch $\\lambda$ [-]')
            ax.set_ylabel('Nominal stress $P$ [MPa]')
        
            
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(outputFolder + '\\elastic_panel.pdf', format='pdf', bbox_inches='tight')
         


def plot_stress(model, trainDs, valDs, outputFolder):
    

    figsize = (22,8)
    
    fig_l, axes_l = plt.subplots(1,2, figsize=figsize)
    axes_l = axes_l.flatten()

    fig_t, axes_t = plt.subplots(1,2, figsize=figsize)
    axes_t = axes_t.flatten()
    
    Ds = [trainDs, valDs]
    Title = ['Training', 'Validation']
    r2_avg = []
    mse_avg = []
        
    for ii, (ds, ax_l, ax_t, t) in enumerate(zip(Ds, axes_l, axes_t, Title)): 
        R2_ = []
        MSE = []
        
        ax_t.set_xlabel(u'Time $t$ [s]')
        ax_t.set_ylabel(u'Nominal stress $P$ [MPa]')
        ax_l.set_xlabel(u'Stretch $\\lambda$ [-]')
        ax_l.set_ylabel(u'Nominal stress $P$ [MPa]')
        
        for jj, (x,y) in enumerate(ds):
            batchSize = x[0].get_shape().as_list()[0]
            stress = model.predict(x)
            
            for kk in range(batchSize):
                F = x[0][kk].numpy()
                lam = F[:,0,0]       
                                
                stress_data = y.numpy()
                R2_.append( R2(stress_data[kk][:,0,0], stress[kk][:,0,0]) )
                MSE.append( ((stress[kk][:,0,0]-stress_data[kk][:,0,0])**2).mean(axis=None) )
                    
                if len(x) == 2:
                    time = x[1][kk].numpy()  
                    ax_t.plot(time, stress[kk][:,0,0], lw=1)
                    n=1
                    ax_t.scatter(time[::n], stress_data[kk][::n,0,0], marker='o', facecolors=None, s=10)
            
        
                ax_l.plot(lam, stress[kk][:,0,0], lw=1)
                n=1
                ax_l.scatter(lam[::n], stress_data[kk][::n,0,0], marker='o', facecolors=None, s=10)                
        
        r2_avg.append(np.mean(R2_))
        mse_avg.append(np.mean(MSE))
    
    subscript = ['tr','val']           
    for axes in [axes_t, axes_l]:
        for ii, ax in enumerate(axes):
            text = 'MSE$_{{{:}}}={:5.4f}$'.format(subscript[ii], mse_avg[ii])
            text += '\n $R^2_{{{:}}}$={:5.4f}'.format(subscript[ii],r2_avg[ii])
            ax.text(0.78, 0.05, text, color='k',transform=ax.transAxes, 
                            bbox=dict(facecolor='none', edgecolor='none'))
            ax.set_title(Title[ii])

    fig_l.savefig(outputFolder+'\\stretch_stress.pdf', format='pdf', bbox_inches="tight")    
    fig_t.savefig(outputFolder+'\\time_stress.pdf', format='pdf', bbox_inches="tight")    



def plot_elastic_stress_biaxial(model, trainDs, valDs, outputFolder):
       
    titles = ['Equi-biaxial', 'Strip-x', 'Strip-y', 'Off-x', 'Off-y']
   
    fig, ax = plt.subplots(1, 5, figsize=(25,6), sharey=True)
    
    R2_ = []
    MSE = []
            
    for jj, (x,y) in enumerate(trainDs):

        stress = model.predict(x)
        stress_data = y.numpy()
        numBatches = x.get_shape().as_list()[0]

        stress_err = np.hstack([stress[:,:,0,0], stress[:,:,1,1]])
        stress_data_err = np.hstack([stress_data[:,:,0,0], stress_data[:,:,1,1]])

        for kk in range(numBatches):
            r2 = R2(stress_data_err[kk], stress_err[kk])
            R2_.append(r2)
            mse = ((stress_err[kk] - stress_data_err[kk])**2).mean(axis=None) 
            MSE.append(mse)

            F = x[kk]
            lam_x = F[:,0,0]
            lam_y = F[:,1,1]
            if titles[kk] == 'Strip-y' or titles[kk] == 'Off-y' or titles[kk] == 'Sub strip-y' or titles[kk] == 'New off-y':
                lam = lam_y
                ax[kk].set_xlabel(r'Stretch $\lambda_y$ [-]')
            elif titles[kk] == 'Strip-x' or titles[kk] == 'Off-x':
                lam = lam_x
                ax[kk].set_xlabel(r'Stretch $\lambda_x$ [-]')
            else:
                lam = lam_x
                ax[kk].set_xlabel(r'Stretch $\lambda$ [-]')
                    
            n=1
            lam = lam.numpy()[::n]
            ax[kk].plot(lam.reshape(-1), stress[kk,::n,0,0], '-', color='tab:red', label=r"$P_{x',pred}$")       
            ax[kk].plot(lam.reshape(-1), stress[kk,::n,1,1], '-', color='tab:blue', label=r"$P_{y',pred}$")   
            ax[kk].plot(lam.reshape(-1), stress_data[kk,::n,0,0], '.', color='tab:red', label=r"$P_{x',data}$")       
            ax[kk].plot(lam.reshape(-1), stress_data[kk,::n,1,1], '.', color='tab:blue', label=r"$P_{y',data}$") 
            
            
            if kk == 0:
                ax[kk].set_ylabel(r'Nominal Stress $P$ [MPa]')
            ax[kk].legend()
            t = titles[kk] + '\n$R^2$={:5.4f} \n MSE={:.3E}'.format(r2, mse)
            ax[kk].set_title(t)
            ax[kk].grid('True')
                      
    plt.tight_layout()
    fig.savefig(outputFolder+'\\elastic_general_biaxial.pdf', format='pdf', bbox_inches="tight")    



def plot_psi_iso(model_Psi_iso, lam_plot, outputFolder, numExtra=0, model_Psi_iso_1=None, model_Psi_iso_2=None, psi_data_iso=None):

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\lambda$ [-]')
    ax.set_ylabel('Strain energy density $\Psi$ [MPa]')
    
    if psi_data_iso is not None:
        psi_model = model_Psi_iso(psi_data_iso[:,0]).numpy()
        ax.plot(psi_data_iso[:,0], psi_model, label='$\Psi_{\mathrm{iso}}$')
        ax.scatter(psi_data_iso[:,0], psi_data_iso[:,1], s=15)
        r2_iso = R2(psi_model.reshape(-1), psi_data_iso[:,1])
        text = r'$R^2_{{\mathrm{{iso}}}}={:5.4f}$'.format(r2_iso)
    else:
        # lam_plot = np.linspace(1,5,n_time_steps)
        if numExtra == 0:
            psi_model = model_Psi_iso(lam_plot)
            ax.plot(lam_plot, psi_model, label='$\Psi_{\mathrm{iso}}$')
        else:
            colors = iter(prop_cycle.by_key()['color'])
            for x,y in trainDs:
                c = next(colors)
                inputs = [x[0], x[2]]
                psi_model = model_Psi_iso(inputs)
                ax.plot(x[0], psi_model, label='$\Psi_{\mathrm{iso}}$', c=c)  
        if model_Psi_iso_1 and model_Psi_iso_2:
            if numExtra == 0:
                c = prop_cycle.by_key()['color'][0]
                psi_model_1 = model_Psi_iso_1(lam_plot)
                psi_model_2 = model_Psi_iso_2(lam_plot)
                ax.plot(lam_plot, psi_model_1, label='$\Psi_{\mathrm{iso},1}$', c=c, linestyle='--')
                ax.plot(lam_plot, psi_model_2, label='$\Psi_{\mathrm{iso},2}$', c=c, linestyle='-.')
            else:
                colors = iter(prop_cycle.by_key()['color'])
                for x,y in trainDs:
                    c = next(colors)
                    inputs = [x[0], x[2]]
                    psi_model_1 = model_Psi_iso_1(inputs)                        
                    psi_model_2 = model_Psi_iso_2(inputs)
                    ax.plot(x[0], psi_model_1, label='$\Psi_{\mathrm{iso},1}$', c=c, linestyle='--')
                    ax.plot(x[0], psi_model_2, label='$\Psi_{\mathrm{iso},2}$', c=c, linestyle='-.')            
    
    if text:
        ax.text(0.7, 0.1, text, color='k',transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='none'))
    ax.legend()
    plt.savefig(outputFolder+'\\psi_iso_plot.pdf', format='pdf', bbox_inches="tight")
                    

def plot_psi_aniso(model_Psi_aniso, lam_plot, phi_all, outputFolder, psi_data_aniso=None):
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\lambda$ [-]')
    ax.set_ylabel('Strain energy density $\Psi$ [MPa]')
    
    if psi_data_aniso is not None:
        R2_ = []
        n = len(psi_data_aniso)//len(phi_all)
        for ii, p in enumerate(phi_all):
            l = psi_data_aniso[ii*n:(ii+1)*n,0]
            psi_data = psi_data_aniso[ii*n:(ii+1)*n,2]
            inps = [l, np.ones_like(l)*p]
            psi_model = model_Psi_aniso(inps).numpy()
            ax.plot(l, psi_model, label=r'$\Psi_{{\mathrm{{ani}}}}$, $\varphi={:.0f}$°'.format(p/np.pi*180.))
            ax.scatter(l, psi_data, s=15)
            r2_aniso = R2(psi_model.reshape(-1), psi_data)
            R2_.append(r2_aniso)
        text = r'$R^2_{{\mathrm{{ani}}}}={:5.4f}$'.format(np.mean(R2_)) 
    
    else:
        for ii, p in enumerate(phi_all):
            inps = [lam_plot, np.ones_like(lam_plot)*p]
            ax.plot(lam_plot, model_Psi_aniso(inps), label=r'$\Psi_{{\mathrm{{ani}}}}$, $\varphi={:.0f}$°'.format(p/np.pi*180.))
            
    if text:
        ax.text(0.7, 0.1, text, color='k',transform=ax.transAxes, 
                        bbox=dict(facecolor='none', edgecolor='none'))
    ax.legend()
    plt.savefig(outputFolder+'\\psi_aniso_plot.pdf', format='pdf', bbox_inches="tight")
    
        

def plot_prony_params(model,
                      trainDs,
                      stretch_range,
                      stretch_rates,
                      nPronyTerms,
                      extrasStruc=None,
                      extras=None, 
                      n=300, 
                      data_coeffs=False,
                      data_times=False,
                      title='Prony Parameters',
                      outputFolder=None):
    
    lw = 1.5
    
    # colors = iter(prop_cycle.by_key()['color'])
    # colors = iter(mpl.colormaps['tab10'](np.linspace(0,1,len(nonzero_idxs)))) 

    layers = [l.name for l in model.layers]
        
    
    ### compute invariants
    
    if 'extra_struc_input' in layers:
        extra_struc_input = model.get_layer('extra_struc_input')
        numExtraStruc = extra_struc_input.input.shape[-1]
    else:
        numExtraStruc = 0
        
    if 'extra_input' in layers:
        extra_input = model.get_layer('extra_input')
        numExtra = len(extras) # extra_input.input.shape[-1]
    else:
        numExtra = 0        
                
    F_input = model.get_layer('F_input').input
    nSteps = F_input.shape[1]
    inputs = [F_input]
    if numExtraStruc > 0:
        inputs.append(extra_struc_input) 
    outputs = [model.get_layer('invars_I').output, model.get_layer('invars_J').output]
    model_invars = tf.keras.models.Model(inputs, outputs)
    
    # deformation gradient
    lam_min = stretch_range[0]
    lam_max = stretch_range[1]
    lam = np.linspace(lam_min,lam_max, nSteps).reshape(1,-1)
    F = defGrad(lam)    

    invars = model_invars(F)
    invars_in = np.concatenate(invars, axis=2)
    
    if 'F_dot_input' in layers:
        rateDependent = True
    else:
        rateDependent = False # if relaxation times are strain rate dependent
        
    if numExtra > 0 and rateDependent:
        print("The plots of the Prony parameters can only be parametrized in terms of either the stretch rate or the extra input, not both")
        exit()        
    
    if rateDependent:
        inputs = [F_input]
        F_dot_input = model.get_layer('F_dot_input').input
        inputs.append(F_dot_input)
        if numExtraStruc > 0:
            inputs.append(extra_struc_input)
        outputs = [model.get_layer('invars_I_dot').output, model.get_layer('invars_J_dot').output, model.get_layer('invar_det_C_dot').output]
        model_invars_dot = tf.keras.models.Model(inputs, outputs)
        
    if numExtra > 0:
        invars_in = [[invars_in, np.ones_like(lam)*e] for e in extras ]

    elif rateDependent:        
        invars_dot = []
        for ii, sr in enumerate(stretch_rates):
            lam_dot = np.ones_like(lam)*sr
            F_dot = defGrad_dot(lam, lam_dot)
            inp = [F,F_dot]
            invars_dot.append(model_invars_dot(inp))
            
        invars_dot = [np.concatenate(invar, axis=2) for invar in invars_dot]
        invars_in = [[invars_in, invar] for invar in invars_dot]


    ### prepare plots
    if numExtra > 0 or rateDependent == True:
        figsize = (13,7)
    else:
        figsize = (15,6)
        
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].set_xlabel(u'Stretch $\\lambda$ [-]')
    ax[0].set_ylabel(u'Relaxation time $\\tau_i$ [s]')
    ax[0].set_xlim([lam_min, lam_max])
    ax[0].set_yscale('log')
    #ax[0].set_ylim([10**(-18), 10**3])
    
    ax[1].set_xlabel(u'Stretch $\\lambda$ [-]')
    ax[1].set_ylabel(u'Relaxation coefficient $g_i$ [-]')
    #ax[1].set_ylim(top=0.9)
    ax[1].set_xlim([lam_min, lam_max])
    
    cmap_0 = plt.get_cmap('coolwarm')


    
    ###--------------------------------   relaxation coefficients   ----------------------------------------
    
    model_g = model.get_layer('model_g_0')
    layer_names = [l.name for l in model_g.layers]
    if any("regularization_layer_g" in ln for ln in layer_names):   
        regularization_layer = model_g.get_layer('regularization_layer_g_0')
        regularization_weights = regularization_layer.weights[0].numpy().squeeze()    
        nonzero_idxs = np.where(regularization_weights > 1e-100)[0]
    else: nonzero_idxs = np.arange(model_g.output.shape.as_list()[-1])

    lam = lam.reshape(-1)
    
    if numExtra == 0 ^ rateDependent == False:
        
        g = np.array(model_g(invars_in)).squeeze()
        g_norm = g/np.sum(g, axis=-1, keepdims=True)            
        #colors = cycle(prop_cycle.by_key()['color'])
        if 0 in nonzero_idxs:
            ax[1].plot(lam, g_norm[:,0], lw=lw, color='k', label='$g_{\\infty}$')
            nonzero_idxs = np.delete(nonzero_idxs, 0)
            colors = iter(prop_cycle.by_key()['color'])             
        else:
            colors = iter(prop_cycle.by_key()['color'])
            
        for ii in nonzero_idxs:
            ax[1].plot(lam, g_norm[:,ii] ,lw=lw, color=next(colors), label='$g_{{{:}}}$'.format(ii))
            
        ax[1].legend()

            
    else:
        if rateDependent:
            bins = len(stretch_rates)
        elif numExtra > 0:
            bins = numExtra
        
        dashes = []
        for ii in range(bins):
            factor = 0.4 + (2-0.4)/(bins-1) * ii
            dash = (factor**1.7*5, factor)
            dashes.append(dash)
        
        for jj, inp in enumerate(invars_in):
            nonzero_idxs = np.where(regularization_weights > 1e-10)[0]
            g = np.array(model_g(inp)).squeeze()
            g_norm = g/np.sum(g, axis=1, keepdims=True)
            #colors = cycle(prop_cycle.by_key()['color'])
            if 0 in nonzero_idxs:
                # cmap_1 = LinearSegmentedColormap.from_list('custom_colormap', ['white', 'k'], N=bins+1)
                # ax[1].plot(lam, g_norm[:,0], color=cmap_1(jj+1), lw=lw, label='$g_{\\infty}$')
                
                ax[1].plot(lam, g_norm[:,0], color='k', dashes=dashes[jj], lw=lw, label='$g_{\\infty}$')
                
                nonzero_idxs = np.delete(nonzero_idxs, 0)
                cmap = ListedColormap(cmap_0(np.linspace(0, 1, len(nonzero_idxs))))
            else:
                cmap = ListedColormap(cmap_0(np.linspace(0, 1, len(nonzero_idxs))))
            for ii, kk in enumerate(nonzero_idxs):
                
                # color = cmap(ii)  # Adjust the color's alpha (transparency)
                # cmap_2 = LinearSegmentedColormap.from_list('custom_colormap', ['white', color], bins+1)
                # color_2 = cmap_2(jj+1)
                # linestyle = '-'
                # ax[1].plot(lam,  g_norm[:,kk], color=color_2, lw=lw, label='$g_{{{:}}}$'.format(ii))
                
                color = cmap(ii)
                linestyle = '--'
                dash_capstyle='round'
                ax[1].plot(lam,  g_norm[:,kk], color=color, lw=lw, linestyle=linestyle, dashes=dashes[jj], dash_capstyle=None, label='$g_{{{:}}}$'.format(kk))
    
    # ### if synthetic data is available
    # if data_coeffs:
    #     relax_coeff_path = data_coeffs
    #     if os.path.exists(relax_coeff_path):
    #         relax_coeff_data = np.load(relax_coeff_path)
    #         nPronyTerms_data = relax_coeff_data.shape[-1]
    #         if phi is not None:
    #             for kk, p in enumerate(phi): 
    #                 relax_coeff_data_tmp = relax_coeff_data[kk]
    #                 g_infy = 1.0 - np.sum(relax_coeff_data_tmp[:,1:], axis=1, keepdims=True)
    #                 relax_coeff_data_tmp = np.hsplit(relax_coeff_data_tmp, nPronyTerms_data)
    #                 lam = relax_coeff_data_tmp[0]
    #                 idxs = np.argsort(lam, axis=None)
    #                 lam_sort = lam[idxs]
    #                 ax[1].plot(lam_sort, g_infy[idxs],lw=lw, color='k', ls='--', label='$g_{{\\infty}}$')            
    #                 for ii in range(nPronyTerms_data-1):
    #                     ax[1].plot(lam_sort,
    #                                relax_coeff_data_tmp[ii+1][idxs],
    #                                color='k',
    #                                ls='--',
    #                                lw=lw,
    #                                label='$g_{{{:},Data}}$'.format(ii+1)
    #                            )
    #         else:
    #             g_infy = 1.0 - np.sum(relax_coeff_data[:,1:], axis=1, keepdims=True)
    #             relax_coeff_data = np.hsplit(relax_coeff_data, nPronyTerms_data)
    #             lam = relax_coeff_data[0]
    #             idxs = np.argsort(lam, axis=None)
    #             lam_sort = lam[idxs]
    #             ax[1].plot(lam_sort, g_infy[idxs], lw=lw, color='k', ls='--', label='$g_{{\\infty}}$')            
    #             for ii in range(nPronyTerms_data-1):
    #                 ax[1].plot(lam_sort,
    #                            relax_coeff_data[ii+1][idxs],
    #                            color='k',
    #                            ls='--',
    #                            lw=lw,
    #                            label='$g_{{{:},Data}}$'.format(ii+1)
    #                        )
                    
                    
    # ax[1].legend(# bbox_to_anchor=(1.04,0.5), 
    #              # loc="center left", 
    #              loc="best", 
    #              fancybox=True,
    #              framealpha=0.8)
    
    # ###--------------------------------   relaxation times   ----------------------------------------
    
    
    # subtract one because the nonzero indices are calculated include g_infy which doesnt have a corresponding relaxation time 
    nonzero_idxs = nonzero_idxs - 1
    
    model_tau = model.get_layer('model_tau_0')
  
    if numExtra == 0 ^ rateDependent == False:        
        inp = lam # x[0]
        relax_time = np.array(model_tau(invars_in)).squeeze()
        relax_time = np.transpose(relax_time)
        # colors = cycle(prop_cycle.by_key()['color'])
        colors = iter(prop_cycle.by_key()['color'])
        for ii in nonzero_idxs:
            ax[0].plot(lam, relax_time[ii],
                        color=next(colors),lw=lw,
                        label='$\\tau_{{{:}}}$'.format(ii+1))
            
        ax[0].legend()
            
        
    else:
        for jj, inp in enumerate(invars_in):
            relax_time = model_tau(inp)
            relax_time = np.array(relax_time).squeeze()
            cmap = ListedColormap(cmap_0(np.linspace(0, 1, len(nonzero_idxs))))
            for ii, kk in enumerate(nonzero_idxs):
                # color = cmap(ii)  # Adjust the color's alpha (transparency)
                # cmap_2 = LinearSegmentedColormap.from_list('custom_colormap', ['white', color], bins+1)
                # color_2 = cmap_2(jj+1)
                # ax[0].plot(lam, relax_time[:,ii], lw=lw, color=color_2, label='$\\tau_{{{:}}}$'.format(ii+1))
                
                color = cmap(ii)
                linestyle = '--'
                dash_capstyle='round'
                ax[0].plot(lam,  relax_time[:,kk], color=color, lw=lw, linestyle=linestyle, dashes=dashes[jj], dash_capstyle=None, label='$\tau_{{{:}}}$'.format(kk))
        
        
    #
        cmap = ListedColormap(cmap_0(np.linspace(0, 1, len(nonzero_idxs))))
        legends = []
        maxwell_legend = False
        legend_lower = True
        
        if maxwell_legend == False:
            legend_elements_g = []    
            legend_elements_tau = []
        else:
            legend_elements_maxwell = []
        
        nonzero_idxs = np.where(regularization_weights > 1e-10)[0]
        if 0 in nonzero_idxs:
            legend_elements_g.append(
                    Patch(color='k', label='$g_{\infty}$')
                    # Line2D([0], [0], linestyle='-', color='tab:black', lw=lw, label='$g_{\infty}$')                    
                )
            nonzero_idxs = np.delete(nonzero_idxs,0)
            
        for ii, jj in enumerate(nonzero_idxs):
            
            if maxwell_legend == False:
                legend_elements_g.append(
                        Patch(color=cmap(ii), label='$g_{{{:}}}$'.format(jj))
                        # Line2D([0], [0], linestyle='-', color=cmap(ii), lw=lw, label='$g_{{{:}}}$'.format(jj))                    
                    )
                legend_elements_tau.append(
                        Patch(color=cmap(ii), label='$\\tau_{{{:}}}$'.format(jj))
                        # Line2D([0], [0], linestyle='-', color=cmap(ii), lw=lw, label='$g_{{{:}}}$'.format(jj))                    
                    ) 
            else:
                legend_elements_maxwell.append(
                        Patch(color=cmap(ii), label=str(jj))
                        # Line2D([0], [0], linestyle='-', color=cmap(ii), lw=lw, label='$g_{{{:}}}$'.format(jj))                    
                    ) 
                
        if maxwell_legend == False:
            l1 = ax[1].legend(handles=legend_elements_g, loc='best')
            l2 = ax[0].legend(handles=legend_elements_tau, loc='best')
            legends.append(l1)
            legends.append(l2)
        else:
            l3 = ax[1].legend(title='Maxwell element', handles=legend_elements_maxwell, loc='upper left', title_fontsize=MEDIUM_SIZE,
                              bbox_to_anchor=(1.05, 1)
                          )
            legends.append(l3)   
            
        #
        if numExtra > 0:
            labels = [str(e*400) for e in extras]
        elif rateDependent:
            labels = [str(sr) for sr in stretch_rates]
            
        if numExtra > 0 or rateDependent == True:        
            legend_elements = []     
            for ii in range(bins):
                legend_elements.append(
                    Line2D([0], [0], linestyle='--', dashes=dashes[ii], color='tab:grey', lw=lw, label=labels[ii])    
                )
            if maxwell_legend == False:
                # l4 = ax[1].legend(title='Stretch rate $\dot{\lambda}$ [s$^{-1}$]', handles=legend_elements, loc='center left', title_fontsize=SMALL_SIZE,
                #                   bbox_to_anchor=(1.05, 0.5)
                #               )
                l4 = plt.figlegend(handles=legend_elements, ncols=bins, loc='upper center', title_fontsize=MEDIUM_SIZE,
                              bbox_to_anchor=(0.5, 0.18), title='Stretch rate $\dot{\lambda}$ [s$^{-1}$]'
                          )
                              
            else:
                l4 = ax[1].legend(title='Stretch rate $\dot{\lambda}$ [s$^{-1}$]', handles=legend_elements, loc='lower left', title_fontsize=SMALL_SIZE,
                                  bbox_to_anchor=(1.05, 0.0)
                              )
    
        #legends.append(l4)
                       
        for l in legends:
            fig.add_artist(l)

            
    # ### synthetic data
    # if data_times:        
    #     relax_time_path = data_times
    #     if os.path.exists(relax_time_path):
    #         relax_time_data = np.load(relax_time_path)
    #         nPronyTerms_data = relax_time_data.shape[-1]
    #         if phi is not None: 
    #             for kk, p in enumerate(phi):
    #                 relax_time_data_tmp = np.hsplit(relax_time_data[kk], nPronyTerms_data)
    #                 lam = relax_time_data_tmp[0]
    #                 idxs = np.argsort(lam, axis=None)
    #                 lam_sort = lam[idxs]
    #                 for ii in range(nPronyTerms_data-1):                
    #                     ax[0].plot(lam_sort, relax_time_data_tmp[ii+1][idxs], lw=lw,color='k', ls='--', label='$\\tau_{{{:},Data}}$'.format(ii+1)) 
    #         else:
    #             relax_time_data = np.hsplit(relax_time_data, nPronyTerms_data)
    #             lam = relax_time_data[0]
    #             idxs = np.argsort(lam, axis=None)
    #             lam_sort = lam[idxs]
    #             for ii in range(nPronyTerms_data-1):                
    #                 ax[0].plot(lam_sort, relax_time_data[ii+1][idxs], lw=lw,color='k', ls='--', label='$\\tau_{{{:},Data}}$'.format(ii+1)) 
                
    # ax[0].legend(#bbox_to_anchor=(1.04,0.5), 
    #              #loc="center left", 
    #              loc="best", 
    #              fancybox=True,
    #              framealpha=0.8)
    
    plt.suptitle(title)
    title = title.replace(" ", "_").lower() 
    #plt.tight_layout()
    if numExtra > 0 or rateDependent == True:
        plt.subplots_adjust(bottom=0.3)
    plt.savefig(outputFolder + '\\{:}.pdf'.format(title), format='pdf', bbox_inches="tight")
    
    
#
###
#

def plot_relaxation_modulus(model, gs, tau, stretch_range, nPronyTerms, phi):
    ### Relaxation modulus
    lam_min = stretch_range[0]
    lam_max = stretch_range[1]
    
    n_lvls = 5
    n_time_steps = 1000
    stretch = np.linspace(lam_min, lam_max, n_lvls)
    t = np.logspace(-4, 4, n_time_steps).reshape(1,-1)
    
    G_I = []
    G_INF = []
    TAU = []
    for ii, p in enumerate(phi):
        inp = [stretch, np.ones_like(stretch)*p]
        g = np.transpose((np.array(gs(inp)).squeeze()))
        g_norm = g/np.sum(g, axis=1, keepdims=True)
        g_i = g_norm[:,1:]
        g_inf = g_norm[:,0:1]

        inp = [stretch, np.ones_like(stretch)*p]
        relax_time = tau(inp)
        
        G_I.append(g_i)
        G_INF.append(g_inf)
        TAU.append(np.hstack(relax_time))
    
    G =[]
    for ii, p in enumerate(phi):
        G_tmp = G_INF[ii]
        for jj in range(nPronyTerms):
            G_tmp = G_tmp + G_I[ii][:,jj:jj+1]*np.exp(-t/TAU[ii][:,jj:jj+1])   
        G.append(G_tmp)
        
    fix, ax = plt.subplots(figsize=(8,6))
    ax.set_xscale('log')
    ax.set_xlabel(u'Time $t$ [s]')
    ax.set_ylim([0,1])
    ax.set_ylabel('Reduced relaxation function $G_{aniso}$ [-]')
    for pp, G_tmp in enumerate(G):
        ls = ['-','--',':','-.',(0, (1, 5))]
        colors = iter(prop_cycle.by_key()['color'])
        for ii, l in enumerate(stretch):
            c = next(colors)
            ax.plot(t.reshape(-1), G_tmp[ii], color=c, linestyle=ls[pp],label=u'$\\varphi$={:.1f}°, $\\lambda=${:.1f}'.format(phi[pp], l) )
    
    ax.legend(title='Stretch level $\\lambda$ [-]', loc='center left')   
#
###
#


###
def plot_delta(model, pathToData, suffix=None, title=None):
    """
    Plots the stress difference between the primary loading curve and the unloading curve 
    as a function of the stretch differnce Delta = lam_{max} - lam
    
    """
    if suffix:
        pathToFile = pathToData + "delta_{:}.json".format(suffix)
    else:
        pathToFile = pathToData + "delta.json"
    f = open(pathToFile,)
    delta_data = json.load(f)
    delta_data = [np.array(d) for d in delta_data]

    if suffix:
        pathToFile = pathToData + "lam_interpolate_{:}.json".format(suffix)
    else:
        pathToFile = pathToData + "lam_interpolate.json"
    f = open(pathToFile,)
    lam_interpolate = json.load(f)
    lam_interpolate = [np.array(d) for d in lam_interpolate]

    if suffix:
        pathToFile = pathToData + "unload_{:}.json".format(suffix)
    else:
        pathToFile = pathToData + "unload.json"        
    f = open(pathToFile,)
    unload = json.load(f)
    unload = [np.array(d) for d in unload]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel(u'Stretch difference $\Delta = \lambda_{max} - \lambda$  [-]')
    ax.set_ylabel(u'Stress difference $\delta$ (damage function $\Gamma$) [MPa]')
    # ax.set_ylim([0,2])
    # ax.set_xlim([0,2])
    
    for ii, dd in enumerate(delta_data):
        delta_stretch_data = dd[:,0] # delta_stretch
        shore_normed = dd[:,1]
        delta_stress_data = dd[:,2] # delta_stress
        ax.scatter(delta_stretch_data, np.flip(delta_stress_data), marker='+')
        # inp = lam_interpolate[ii]
        # gamma = model_dmgIso(inp)
        # delta = lam_max[ii] - lam_interpolate[ii]
        # ax.plot(delta, gamma, color=color_cycle[ii], label="$\lambda_{{max}}$={:}".format(lam_max[ii]))
        lam = unload[ii][:,0]
        lam_max = np.max(lam)
        delta = lam_max - lam
        x = (lam, np.ones_like(lam)*shore_normed[0])
        gamma = model(x)
        ax.plot(delta, gamma, label="$\lambda_{{max}}$={:}".format(np.around(lam_max, decimals=1)))
        
    ax.legend()
    ax.grid(True)
    if suffix and title:
        title = title.format(suffix)
        ax.set_title(title)
    elif title:
        ax.set_title(title)
    
    figPath = pathToData + 'stress_difference_{:}'.format(suffix)
    fig.savefig(figPath, format='pdf')
    
    

def plot_hossain(model, pathToData, outputFolder, rateDependent=False):

    path = outputFolder + '\\prediction'
    if not os.path.exists(path):
        os.makedirs(path)

    LAM = [3.0, 1.5, 2.0, 2.5]
    LAM_DOT = [0.01, 0.03, 0.05]
    
    fig_p, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    captions = ['(a)','(b)','(c)','(d)']
    colors = ['tab:blue', 'tab:green', 'tab:orange']
        
    for ii, lam in enumerate(LAM):
    
        axes = axes.flatten()
        fig, ax = plt.subplots(figsize=(8.5, 7))     
        
        xlabel = 'Stretch $\\lambda$ [-]'
        ylabel = 'Nominal stress $P$ [kPa]'
        axes[ii].set_xlabel(xlabel)
        axes[ii].set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        MSE = []
        NRMSE = []
        R2_ = []
    
        for jj, lam_dot in enumerate(LAM_DOT):
    
            if lam == 3.0 and lam_dot == 0.03:
                continue
                        
            # interpolated training data
            data = np.load(pathToData + '\\stretch_{:}\\{:}.npy'.format(lam, lam_dot))   
            stretch, time, stress = data[:,0], data[:,1], data[:,2]
            if rateDependent:
                stretch_rate = data[:,3]
                F_dot = defGrad_dot(stretch.reshape(1,-1), stretch_rate.reshape(1,-1))
            F = defGrad(stretch.reshape(1,-1))
            time = np.expand_dims(time,0)
            stress_out = np.expand_dims(stress, 0)
            batch_size = len(F)

            extra = np.ones_like(time)
            
            if rateDependent:
                inps = (F, time, F_dot) # , extra)
            else:
                inps = (F, time) # , extra)
            outs = (stress_out)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            stress_pred = stress_pred[0,:,0,0]
            
            r2 = R2(stress_pred, stress)
            mse = ((stress_pred - stress)**2).mean(axis=None)
            nrmse = np.sqrt(mse)/(np.max(stress) - np.min(stress))
            R2_.append(r2)
            MSE.append(mse)
            NRMSE.append(nrmse)
           
            
            axes[ii].plot(stretch, stress_pred, color=colors[jj], lw=2.)
            ax.plot(stretch, stress_pred, color=colors[jj], lw=2.5)
            
            # save vCANN prediction to file
            pred_data = np.vstack([stretch, time.reshape(-1), stress_pred])
            file = path + '\\lam_{}_lam_dot_{}.npy'.format(lam, lam_dot)
            np.save(file, pred_data)
            
            # experimental data
            data = np.loadtxt(pathToData + '\\stretch_{:}\\{:}.txt'.format(lam, lam_dot), delimiter=',')   
            stretch, stress = data[:,0], data[:,1]
            
            if lam == 1.5:
                label='${:}$'.format(lam_dot)
            else:
                label=''

            label='${:}$, {:3.2f}, {:3.2f}'.format(lam_dot,  r2, mse)
            axes[ii].scatter(stretch, stress, s=25, facecolors='none', edgecolor=colors[jj], label=label)
            axes[ii].grid(True)
            ax.scatter(stretch, stress, s=30, facecolors='none', edgecolor=colors[jj], label=label)
            ax.grid(True)
        
        axes[ii].legend(title='$\\dot{\\lambda}$ [s$^{-1}$]') #, $R^2$ [-], $\\varepsilon$ [-]')
        axes[ii].text(.5, -.21, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
        
        ax.legend(title='$\\dot{\\lambda}$ [s$^{-1}$]') #, $R^2$ [-], $\\varepsilon$ [-]')
                
        if ii == 0:
            t = 'Training' # \n $R^2$={:5.4f} \n MSE={:.3E}'.format(r2, mse)
            axes[ii].set_title(t)
            ax.set_title(t)
        else:
            t = 'Validation' # \n $R^2$={:5.4f} \n MSE={:.3E}'.format(r2, mse)
            axes[ii].set_title(t)
            ax.set_title(t)

        fig.tight_layout()        
        fig.savefig(outputFolder + '\\training_validation_{:}.pdf'.format(ii+1), format='pdf', bbox_inches='tight')


    fig_p.subplots_adjust(hspace=0.4)
    fig_p.savefig(outputFolder + '\\training_validation_panel.pdf', format='pdf', bbox_inches='tight')
    # plt.close('all')


def plot_liao_thermo_1(model, pathToData, outputFolder):

    ### training    

    THETA_TRAIN = np.load(pathToData + 'data_split\\theta_train.npy')
    LAM_TRAIN = np.load(pathToData + 'data_split\\lam_train.npy')
    LAM_DOT_TRAIN = np.load(pathToData + 'data_split\\lam_dot_train.npy') 

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
                
                fname = pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot)
                
                if not os.path.isfile(fname):
                    continue
                
                if lam_dot == 0.05:
                    continue
                if theta == 60.0:
                    continue
                # if lam_dot == 0.03 or lam_dot == 0.05:
                #     continue

                data = np.load(pathToData + '\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot))   
                stretch, time, temp, stress = data[:,0], data[:,1], data[:,2], data[:,3]
                
                batch_size = len(stretch)
                inps = (stretch, time, temp)
                outs = (stress)
                ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
                stress_pred = model.predict(ds)

                if lam == 4:
                    axes[ii].plot(stretch, stress_pred, color=c, linewidth=1, label=lam_dot)          
                else:
                    axes[ii].plot(stretch, stress_pred, color=c, linewidth=1)
                
                data = np.loadtxt(pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot), delimiter=',')   
                stretch, stress = data[:,0], data[:,1]
                axes[ii].scatter(stretch, stress, s=15, color=c)
                   
                axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
                axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ $[s^{-1}]$')
                axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))
                
    fig.suptitle('Training data')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(outputFolder + '\\training_data.pdf', format='pdf') 

    ### Validation

    THETA_VAL = np.load(pathToData + 'data_split\\theta_val.npy')
    LAM_VAL = np.load(pathToData + 'data_split\\lam_val.npy')
    LAM_DOT_VAL = np.load(pathToData + 'data_split\\lam_dot_val.npy') 
       
    fig, axes = plt.subplots(3, 2, figsize=(14, 26))
    axes = axes.flatten()
    
    for ii, theta in enumerate(THETA_VAL):
        axes[ii].set_xlabel('Stretch $\\lambda$ [-]')
        axes[ii].set_ylabel('Nominal stress $P$ [MPa]')
        for lam in LAM_VAL:
            colors = iter(prop_cycle.by_key()['color'])
            for jj, lam_dot in enumerate(LAM_DOT_VAL):
                c = next(colors)
                
                fname = pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot)
                if not os.path.isfile(fname):
                    continue
                
                if theta !=60:
                    if lam_dot == 0.03:
                        continue
                    if lam == 4 and lam_dot !=0.05:
                        continue

                # if lam_dot == 0.1 and lam == 4:
                #     continue
                    
                data = np.load(pathToData + '\\{:}\\{:}_{:}.npy'.format(theta, lam, lam_dot))   
                stretch, time, temp, stress = data[:,0], data[:,1], data[:,2], data[:,3]
                
                batch_size = len(stretch)
                inps = (stretch, time, temp)
                outs = (stress)
                ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
                stress_pred = model.predict(ds)
                axes[ii].plot(stretch, stress_pred, color=c, linewidth=1, label=lam_dot)
                
                data = np.loadtxt(pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, lam, lam_dot), delimiter=',')   
                stretch, stress = data[:,0], data[:,1]
                axes[ii].scatter(stretch, stress, s=15, color=c)
                       
                axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
                axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ $[s^{-1}]$')
                axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))
    
    fig.suptitle('Validation data')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(outputFolder + '\\validation_data.pdf', format='pdf') 
    
          

def plot_liao_thermo(model, pathToData, outputFolder, rateDependent):

    THETA = [0.0, 20.0, 40.0, 60.0, 80.0]
    LAM_DOT = [0.1]

    fig_p, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    captions = ['(a)','(b)','(c)','(d)']


    # colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel('Stretch $\\lambda$ [-]')
    ax.set_ylabel('Nominal stress $P$ [MPa]')
    
    axes[0].set_xlabel('Stretch $\\lambda$ [-]')
    axes[0].set_ylabel('Nominal stress $P$ [MPa]')    

    for ii, lam_dot in enumerate(LAM_DOT):

        # c=colors[ii]
        for jj, theta in enumerate(THETA):
                                 
            data = np.load(pathToData + '\\{:}\\{:}.npy'.format(theta, lam_dot))   
            stretch, time, temp, stress = data[:,0], data[:,1], data[:,2], data[:,3]
            
            batch_size = len(stretch)
            inps = (stretch, time, temp)
            outs = (stress)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            ax.plot(stretch, stress_pred, lw=1.)
            axes[0].plot(stretch, stress_pred, lw=1.)
            
            label='$\\Theta={:.0f}$ °C'.format(theta)
            data = np.loadtxt(pathToData + '\\{:}\\{:}.txt'.format(theta, lam_dot), delimiter=',')   
            stretch, stress = data[:,0], data[:,1]
            ax.scatter(stretch, stress, s=15, label=label)
            axes[0].scatter(stretch, stress, s=15, label=label)
            
    ax.set_title('Training')
    ax.legend(title='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$'.format(lam_dot))
    fig.savefig(outputFolder + '\\training_data.pdf', format='pdf')    

    axes[0].set_title('Training')
    axes[0].legend(title='$\\dot{{\\lambda}}$ = {:} $s^{{-1}}$'.format(lam_dot))
    axes[0].text(.5, -.2, captions[0], horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    
    
    # validation data for 0 and 80 degrees
    THETA = [0.0, 80.0]
    LAM = [2.0, 3.0]

    for ii, theta in enumerate(THETA):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel('Stretch $\\lambda$ [-]')
        ax.set_ylabel('Nominal stress $P$ [MPa]')
        
        axes[ii+1].set_xlabel('Stretch $\\lambda$ [-]')
        axes[ii+1].set_ylabel('Nominal stress $P$ [MPa]')
            
        # c=colors[ii]
        for lam in LAM:
                                 
            data = np.load(pathToData + '\\validation\\{:}\\{:}.npy'.format(theta, lam))   
            stretch, time, temp, stress = data[:,0], data[:,1], data[:,2], data[:,3]
            
            batch_size = len(stretch)
            inps = (stretch, time, temp)
            outs = (stress)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            ax.plot(stretch, stress_pred, lw=1.)
            axes[ii+1].plot(stretch, stress_pred, lw=1.)
            
            data = np.loadtxt(pathToData + '\\validation\\{:}\\{:}.txt'.format(theta, lam), delimiter=',')   
            stretch, stress = data[:,0], data[:,1]
            ax.scatter(stretch, stress, s=15)
            axes[ii+1].scatter(stretch, stress, s=15)
            
        axes[ii+1].legend(title='$\\Theta={:.0f}$ °C, $\\dot{{\\lambda}}$ = {:} $s^{{-1}}$'.format(theta, lam_dot), frameon=False)
        axes[ii+1].set_title('Validation')
        axes[ii+1].text(.5, -.2, captions[ii+1], horizontalalignment='center', verticalalignment='center', transform=axes[ii+1].transAxes)

        ax.set_title('Validation')
        ax.legend(title='$\\Theta={:.0f}$ °C'.format(theta), frameon=False)
        fig.savefig(outputFolder + '\\validation_data_{:.0f}.pdf'.format(theta), format='pdf')

    # validation data for 10 degrees
    THETA = [10.0]
    LAM = [3.0, 4.0]

    for ii, theta in enumerate(THETA):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel('Stretch $\\lambda$ [-]')
        ax.set_ylabel('Nominal stress $P$ [MPa]')
        
        axes[3].set_xlabel('Stretch $\\lambda$ [-]')
        axes[3].set_ylabel('Nominal stress $P$ [MPa]')
        
        # c=colors[ii]
        for lam in LAM:
                                 
            data = np.load(pathToData + '\\validation\\{:}\\{:}.npy'.format(theta, lam))   
            stretch, time, temp, stress = data[:,0], data[:,1], data[:,2], data[:,3]
            
            batch_size = len(stretch)
            inps = (stretch, time, temp)
            outs = (stress)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            ax.plot(stretch, stress_pred, lw=1.)
            axes[3].plot(stretch, stress_pred, lw=1.)
            
            data = np.loadtxt(pathToData + '\\validation\\{:}\\{:}.txt'.format(theta, lam), delimiter=',')   
            stretch, stress = data[:,0], data[:,1]
            ax.scatter(stretch, stress, s=20)
            axes[3].scatter(stretch, stress, s=20)
            
        axes[3].set_title('Validation')
        axes[3].legend(title='$\\Theta={:.0f}$ °C, $\\dot{{\\lambda}}$ = {:} $s^{{-1}}$'.format(theta, lam_dot), frameon=False)
        axes[3].text(.5, -.2, captions[3], horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)

        
        ax.set_title('Validation')
        ax.legend(title='$\\Theta={:.0f}$ °C'.format(theta), frameon=False)
        
        
    fig.savefig(outputFolder + '\\validation_data_{:.0f}.pdf'.format(theta), format='pdf')
    
    fig_p.subplots_adjust(hspace=0.4)
    fig_p.savefig(outputFolder + '\\training_validation_panel.pdf', format='pdf', bbox_inches='tight')
        
    

def plot_vhb_4905(model, pathToData, outputFolder, rateDependent):

    ### training    

    THETA_TRAIN = np.load(pathToData + 'data_split\\theta_train.npy')
    LAM_TRAIN = np.load(pathToData + 'data_split\\lam_train.npy')
    LAM_DOT_TRAIN = np.load(pathToData + 'data_split\\lam_dot_train.npy') 

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
                
                fname = pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot)
                
                if not os.path.isfile(fname):
                    continue
                
                # if lam_dot == 0.05:
                #     continue
                # if theta == 60.0:
                #     continue
                # if lam_dot == 0.03 or lam_dot == 0.05:
                #     continue

                ### as in the paper's main part
                if lam_dot == 0.05:
                    continue

                data = np.load(pathToData + '\\{:}\\{:}_{:}.npy'.format(theta, int(lam), lam_dot))   
                stretch, time, temp, P11 = data[:,0], data[:,1], data[:,2], data[:,3]
                
                if rateDependent:
                    stretch_rate = data[:,4]
                    F_dot = defGrad_dot(stretch.reshape(1,-1), stretch_rate.reshape(1,-1))

                F = defGrad(stretch.reshape(1,-1))
                time = np.expand_dims(time,0)
                stress_out = np.expand_dims(P11, 0)
                temp = np.expand_dims(temp,0)

                if rateDependent:
                    inps = (F, time, F_dot, temp)
                else:
                    inps = (F, time, temp)
                outs = (stress_out)

                batch_size = len(F)
                ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
                stress_pred = model.predict(ds)
                stress_pred = stress_pred[0,:,0,0]

                if lam == 4:
                    axes[ii].plot(stretch, stress_pred, color=c, marker='.', linewidth=1, label=lam_dot)          
                else:
                    axes[ii].plot(stretch, stress_pred, color=c, marker='.', linewidth=1)
                
                data = np.loadtxt(pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot), delimiter=',')   
                stretch, stress = data[:,0], data[:,1]*100
                axes[ii].scatter(stretch, stress, s=15, color=c)
                   
                axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
                axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ $[s^{-1}]$')
                axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))
                
    fig.suptitle('Training data')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(outputFolder + '\\training_data.pdf', format='pdf') 

    ### Validation

    THETA_VAL = np.load(pathToData + 'data_split\\theta_val.npy')
    LAM_VAL = np.load(pathToData + 'data_split\\lam_val.npy')
    LAM_DOT_VAL = np.load(pathToData + 'data_split\\lam_dot_val.npy') 
       
    fig, axes = plt.subplots(3, 2, figsize=(14, 26))
    axes = axes.flatten()
    
    for ii, theta in enumerate(THETA_VAL):
        axes[ii].set_xlabel('Stretch $\\lambda$ [-]')
        axes[ii].set_ylabel('Nominal stress $P$ [MPa]')
        for lam in LAM_VAL:
            colors = iter(prop_cycle.by_key()['color'])
            for jj, lam_dot in enumerate(LAM_DOT_VAL):
                c = next(colors)
                
                fname = pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot)
                if not os.path.isfile(fname):
                    continue
                
                # if theta !=60:
                #     if lam_dot == 0.03:
                #         continue
                #     if lam == 4 and lam_dot !=0.05:
                #         continue

                # if lam_dot == 0.1 and lam == 4:
                #     continue

                ### as in the paper's main part
                if lam_dot == 0.03:
                    continue
                if lam == 4 and lam_dot !=0.05:
                    continue
                    
                data = np.load(pathToData + '\\{:}\\{:}_{:}.npy'.format(theta, int(lam), lam_dot))   
                stretch, time, temp, P11 = data[:,0], data[:,1], data[:,2], data[:,3]
                
                if rateDependent:
                    stretch_rate = data[:,4]
                    F_dot = defGrad_dot(stretch.reshape(1,-1), stretch_rate.reshape(1,-1))

                F = defGrad(stretch.reshape(1,-1))
                time = np.expand_dims(time,0)
                stress_out = np.expand_dims(P11, 0)
                temp = np.expand_dims(temp,0)

                if rateDependent:
                    inps = (F, time, F_dot, temp)
                else:
                    inps = (F, time, temp)
                outs = (stress_out)

                batch_size = len(F)
                ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
                stress_pred = model.predict(ds)
                stress_pred = stress_pred[0,:,0,0]
                
                axes[ii].plot(stretch, stress_pred, color=c, marker='.',linewidth=1, label=lam_dot)
                
                data = np.loadtxt(pathToData + '\\{:}\\{:}_{:}.txt'.format(theta, int(lam), lam_dot), delimiter=',')   
                stretch, stress = data[:,0], data[:,1]*100
                axes[ii].scatter(stretch, stress, s=15, color=c)
                       
                axes[ii].text(.5, -.2, captions[ii], horizontalalignment='center', verticalalignment='center', transform=axes[ii].transAxes)
                axes[ii].legend(title='Stretch rate $\\dot{\\lambda}$ $[s^{-1}]$')
                axes[ii].set_title('$\\Theta = {:}$ °C'.format(theta))
    
    fig.suptitle('Validation data')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(outputFolder + '\\validation_data.pdf', format='pdf') 


    
def plot_calvo(model, trainDs, valDs, outputFolder):
    
    fig, axes = plt.subplots(1, 2, figsize=(17,6))

    scale = [1,1.04,1.,1,1,1]
    
    titles = ['Training', 'Validation']
    
    for ii, ds in enumerate([trainDs, valDs]):
        
        axes[ii].set_xlabel(u'Time $t$ [s]')
        axes[ii].set_ylabel(u'Reduced relaxation function $G$ [-]')
      
        axes[ii].set_xlim([0,1800])
        axes[ii].set_ylim([0.4,1])  
        
        #axes[ii].set_title('{:} data'.format(titles[ii]))
        
        for jj, (x,y) in enumerate(ds): 
            batchSize = x[0].get_shape().as_list()[0]
            F = x[0]
            time = x[1]
            stress_data = y
            stress = model.predict(x)
            
            for kk in range(batchSize):
                lam = F[kk,:,0,0].numpy()
                t = time[kk].numpy()
                s = stress[kk,:,0,0]
                s_d = stress_data[kk,:,0,0].numpy()
      
                l = '$\lambda = {:}$'.format(lam[0])
                axes[ii].plot(t, s, lw=1.5, label=l)
            
                n=3
                stress_scale = (s_d - 1)/scale[jj] + 1
                # if ii ==0:
                #     axes[ii].scatter(time[::n], stress_scale[::n], marker='o', s=20)
                # else:
                axes[ii].scatter(t[::n], s_d[::n], marker='o', s=20)
                
        axes[ii].legend(fancybox=True, framealpha=0.8)
    # axes[0].plot(time, stress, lw=1.5, color='k')        
    fig.savefig(outputFolder+'\\strain_dependent_relaxation.pdf', format='pdf', bbox_inches="tight")  
    
    
    ### plot single figure
    fig, ax = plt.subplots(figsize=(8,6))

    scale = [1,1.04,1.,1,1,1]
    
    titles = ['Training', 'Validation']
    
    for ii, ds in enumerate([trainDs, valDs]):
        
        ax.set_xlabel(u'Time $t$ [s]')
        ax.set_ylabel(u'Reduced relaxation function $G$ [-]')
      
        ax.set_xlim([0,1800])
        ax.set_ylim([0.4,1])  
        
        ax.set_title('{:} data'.format(titles[ii]))
        
        for jj, (x,y) in enumerate(ds): 
            batchSize = x[0].get_shape().as_list()[0]
            F = x[0]
            time = x[1]
            stress_data = y
            stress = model.predict(x)
            
            for kk in range(batchSize):
                lam = F[kk,:,0,0].numpy()
                t = time[kk].numpy()
                s = stress[kk,:,0,0]
                s_d = stress_data[kk,:,0,0].numpy()
      
                l = '$\lambda = {:}$'.format(lam[0])
                ax.plot(t, s, lw=1.5, label=l)
            
                n=3
                stress_scale = (s_d - 1)/scale[jj] + 1
                # if ii ==0:
                #     axes[ii].scatter(time[::n], stress_scale[::n], marker='o', s=20)
                # else:
                ax.scatter(t[::n], s_d[::n], marker='o', s=20)
            
        ax.legend(fancybox=True, framealpha=0.8)
      
    fig.savefig(outputFolder+'\\strain_dependent_relaxation_overview.pdf', format='pdf', bbox_inches="tight")  
    
    
    
    
def plot_synthetic(model, Ds, Steps, outputFolder):
    figsize = (14,18)
    
    fig_l, axes_l = plt.subplots(2,1, figsize=figsize)
    axes_l = axes_l.flatten()

    fig_t, axes_t = plt.subplots(2,1, figsize=figsize)
    axes_t = axes_t.flatten()
    
    Title = ['Training', 'Validation']
    
    for ii, (ds, ax_l, ax_t, t, steps) in enumerate(zip(Ds, axes_l, axes_t, Title, Steps)): 
        
        lam_max = 1.
        lam_min = 1.
        for jj, (x,y) in enumerate(ds):
            
            lam = x[0].numpy()
            time = x[0].numpy()
            if np.max(lam) > lam_max:
                lam_max = np.max(lam)
            if np.min(lam) < lam_min:
                lam_min = np.min(lam)
                
            stress = model.predict(x, batch_size=steps).reshape(-1)
            stress_data = y.numpy().reshape(-1)
    
            l =  str(x[2].numpy()[0,0]/np.pi*180.)
            
            time = x[1]
            ax_t.set_xlabel(u'Time $t$ [s]')
            ax_t.set_ylabel(u'Nominal stress $P$ [MPa]')
            ax_t.plot(time, stress, lw=1, label=l)
            n=4
            ax_t.scatter(time[::n], stress_data[::n], marker='o', facecolors=None, s=10)

            
            ax_l.set_xlabel(u'Stretch $\\lambda$ [-]')
            ax_l.set_ylabel(u'Nominal stress $P$ [MPa]')
            ax_l.plot(lam, stress, lw=1, label=l)
            n=2
            ax_l.scatter(lam[::n], stress_data[::n], marker='o', facecolors=None, s=10)                

            fs = 14
            for ax in [ax_t, ax_l]:
                bbox = ax.get_position()
                x0 = bbox.x0
                y0 = bbox.y0
                width_0 = bbox.width 
                height_0 = bbox.height
                x = x0 + 0.10*width_0
                y = y0 + 0.78*height_0
                width = 0.38*width_0
                height = 0.15*height_0
                sub_ax = ax.figure.add_axes([x, y, width, height]) 
                sub_ax.set_ylabel('Stretch $\lambda$ [-]', fontsize=fs)
                sub_ax.set_xlabel('Time $t$ [s]', fontsize=fs)
                sub_ax.tick_params(axis='both', which='major', labelsize=fs)
                sub_ax.tick_params(axis='both', which='minor', labelsize=fs)
                sub_ax.plot(time, lam)
            
        for ax in [ax_t, ax_l]:        
            ax.set_title(t)
            ax.set_title(t)
            
        ax_t.legend(#bbox_to_anchor=(0.05,0.95),
                      title='Fiber angle $\\varphi$ [°]',
                      loc="lower left", 
                      fancybox=True,
                      framealpha=0.8,
                      )

        ax_l.legend(#bbox_to_anchor=(0.05,0.95),
                      title='Fiber angle $\\varphi$ [°]',
                      loc="lower right", 
                      fancybox=True,
                      framealpha=0.8,
                      )
        
    fig_l.savefig(outputFolder+'\\final_model_stretch.pdf', format='pdf', bbox_inches="tight")    
    fig_t.savefig(outputFolder+'\\final_model_time.pdf', format='pdf', bbox_inches="tight")    

    
    
def plot_Linz(model, pathToData, outputFolder):
    
    titles = ['Training', 'Validation']
    prefix = ['train', 'valid']
    
    fig, axes = plt.subplots(1,2, figsize=(15,6))
    axes = axes.flatten()
    
    for ii in range(2):
        axes[ii].set_xlabel(u'Stretch $\lambda$ [-]')
        axes[ii].set_ylabel(u'Nominal stress $P$ [MPa]')
        axes[ii].set_title('{:} data'.format(titles[ii]))
          
        folder = pathToData + '{}_data'.format(prefix[ii])
        files = os.listdir(folder)
        files = natsorted(files)
        for f in files:
        
            # interpolated training data
            data = np.load(os.path.join(folder,f))   
            lam, time, lam_dot, stress = data[:,0], data[:,1], data[:,2], data[:,3]
            
            # interpolated training data

            
            F = defGrad(lam.reshape(1,-1))
            F_dot = defGrad_dot(lam.reshape(1,-1), lam_dot.reshape(1,-1))
            time = np.expand_dims(time, 0)
            stress = np.expand_dims(stress, 0)
            batch_size = len(F)
            
            # inps = (F, time, F_dot)
            inps = (F, time, lam_dot.reshape(1,-1))
            outs = (stress,)
            ds = tf.data.Dataset.from_tensor_slices((inps, outs)).batch(batch_size)
            stress_pred = model.predict(ds)
            stress_pred = stress_pred[0]*lam
            
            label =  str(lam_dot[0]*400.0)
            axes[ii].plot(lam, stress_pred, lw=1.5, label=label)
            
            n=4
            axes[ii].scatter(lam[::n], stress[0,::n], marker='o', facecolors=None, s=20)
        
        axes[ii].legend(#bbox_to_anchor=(1.05,0.5),
                        title=u'Stretch rate $\\dot{\\lambda}$ [s$^{-1}$]',
                        loc='best',#"center left", 
                        fancybox=True,
                        framealpha=0.8,
                        )

    fig.tight_layout()
    fig.savefig(outputFolder+'\\final_model.pdf', format='pdf', bbox_inches="tight")  



    # ###
    # fig, ax = plt.subplots(figsize=(8,6))
    
    # ax.set_xlabel(u'Stretch $\\lambda$ [-]')
    # ax.set_ylabel(u'Stress difference $P$ [MPa]')
    
    # for ii, ds in enumerate([trainDs, valDs]):
    #     axes[ii].set_xlabel(u'Stretch $\lambda$ [-]')
    #     axes[ii].set_ylabel(u'Nominal stress $P$ [MPa]')
        
    #     for jj, (x,y) in enumerate(ds):
    #         lam = x[0].numpy()
    #         stress = model.predict(x, batch_size=steps).reshape(-1)
    #         stress_data = y.numpy().reshape(-1)
    #         l =  str(x[2].numpy()[0]*400.0)
    #         n=2
    #         delta_stress = stress_data - stress
    #         ax.scatter(lam[::n], delta_stress[::n], marker='o', facecolors=None, s=20, label=l)

    #     ax.legend(#bbox_to_anchor=(0.05,0.95),
    #               title=u'$\\dot{\\lambda}$ [s$^{-1}$]',
    #               loc="best", 
    #               fancybox=True,
    #               framealpha=0.8,
    #               )
        
    # fig.savefig(outputFolder+'\\final_model_delta.pdf', format='pdf', bbox_inches="tight")    
      
