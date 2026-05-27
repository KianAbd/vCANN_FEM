# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:12:35 2025

@author: Kian
"""

import tensorflow as tf

import ContinuumMechanics as CM
import subANNs
import layers
import utils


def build_model(nSteps,
                numTens,
                numDir,
                nMaxwell,
                numExtra,
                numExtraStruc,
                uncoupled,
                rateDependent,
                layer_size_psi,
                activations_psi,
                layer_size_dir,
                activations_dir,
                layer_size_w,
                activations_w,
                layer_size_tau,
                activations_tau,
                layer_size_psi_a, 
                activations_psi_a,
                tau = [1e-3, 1e3],
                lambda_maxwell=0.001,
                incomp=True,
                visco=False,
                tf_float='float64'):

    ### Deformation measures
    # current configuration
    F = tf.keras.layers.Input(shape=(nSteps,3,3,), name='F_input') # INPUT
    F_dot = tf.keras.layers.Input(shape=(nSteps,3,3,), name='F_dot_input') # INPUT
    batchSize = tf.shape(F)[0]

    C = tf.keras.layers.Lambda(lambda F: CM.ten2_C(F), name='C' )(F)
    C_bar = tf.keras.layers.Lambda(lambda C: CM.ten2_C_bar(C), name='C_bar')(C)
    C_dot = tf.keras.layers.Lambda(lambda x: CM.ten2_C_dot(x[0],x[1]), name='C_dot' )([F,F_dot])
    C_bar_dot = tf.keras.layers.Lambda(lambda x: CM.deviatoric_projection_transposed(x[0], x[1]), name='C_bar_dot' )([C_dot, C])
    
    # Determination of the reference configuration (no reference for the invariants based on C_dot necessary)
    F_ref = tf.keras.layers.Lambda(lambda F: CM.ten2_F_ref(F), name='F_ref')(F)    # DO NOT use output_shape=(None,tf.shape(F)[1],3,3) as argument to the lambda layer. Will cause massive problems when saving / serialzing
    C_ref = tf.keras.layers.Lambda(lambda F: CM.ten2_C(F), name='C_ref')(F_ref)
    C_bar_ref = tf.keras.layers.Lambda(lambda C: CM.ten2_C_bar(C), name='C_bar_ref')(C_ref)   
    
    ### other extra feature inputs which affect the material properties (e.g. temperature, filler content, ...)
    if numExtra == 0:
        extra_in = []
    else:
        extra_in = tf.keras.layers.Input(shape=(nSteps,numExtra,), name='extra_input') # INPUT    
       
    ### Directions and structure tensors
    # isotropic material
    if numDir == 0:
        extra_struc_in = []
        dir_model = []
        w_model = []
        
        dir = [] # we do not need directions (and hence their sub-ANN) at all
        w = tf.ones([batchSize,nSteps,numTens,1], dtype=tf_float) # we do not need a sub-ANN to get the weights
        shaper = batchSize*tf.constant([1,0,0,0,0]) + tf.constant([0,0,1,1,1]) + nSteps*tf.constant([0,1,0,0,0])
        L = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.zeros([3,3], dtype=tf_float),0),0),0), shaper)
        
    # anisotropic material; fiber directions and weights of the structural tensors do not depend on any additional input
    elif numDir != 0 and numExtraStruc == 0:
        extra_struc_in = []
        
        model_dir = CM.dirModelOrtho(numTens, numDir, name='model_dir')
        dir = model_dir(F)  # (?,nSteps,numDir,3)
        model_weight = CM.weightModelOrtho(numTens, numDir, name='model_w')
        w = model_weight(F) # (?,nSteps,numTens,numDir+1)
        L = tf.keras.layers.Lambda(lambda dir: CM.ten2_L(dir), name='L')(dir) # (?,nSteps,numDir,3,3)
        
    # anisotropic material; fiber directions and weights of the structural tensors depend on additional input
    elif numDir != 0 and numExtraStruc != 0:
        extra_struc_in = tf.keras.layers.Input(shape=(nSteps, numExtraStruc,), name='extra_struc_input') # INPUT

        # Create model from direction sub ANN
        dir_ANNs = []
        for ii in range(numDir):
            dir_ann = subANNs.dir_subANN(extra_struc_in, layer_size_dir, activations_dir, str(ii+1)) # (?,nSteps,3)
            dir_ANNs.append(dir_ann)

        dir_ANNs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-2), name='stack_dir')(dir_ANNs) 
        dir_model = tf.keras.models.Model(inputs=extra_struc_in, outputs=dir_ANNs, name='model_dir') # (?,nSteps,numDir,3)

        # Create model from weights sub ANN
        w_ANNs = []
        for ii in range(numTens):
            w_ann = subANNs.w_subANN(extra_struc_in, layer_size_w, activations_w, numDir, str(ii+1))  # (?,nSteps,numDir+1)
            w_ANNs.append(w_ann)

        w_ANNs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-2), name='stack_w')(w_ANNs)
        w_model = tf.keras.models.Model(inputs=extra_struc_in, outputs=w_ANNs, name='model_w') # (?,nSteps,numTens,numDir+1)
        
        dir = dir_model(extra_struc_in) # (?,nSteps,numDir,3)
        w = w_model(extra_struc_in) # (?,nSteps,numTens,numDir+1) ???
        L = tf.keras.layers.Lambda(lambda dir: CM.ten2_L(dir), name='L')(dir) # (?,nSteps,numDir,3,3)
    
    ### Generalized structure tensors
    H = tf.keras.layers.Lambda(lambda x: CM.ten2_H(x[0], x[1], x[2], x[3], x[4]), name='H')([L, w, nSteps, numDir, numTens])      
    
    ### Invariants
    if incomp:
        # isochoric invariants
        I = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]), name='invars_I')([C_bar,H,numTens]) # [?,nSteps,numTens]
        J = tf.keras.layers.Lambda(lambda x: CM.invariants_J_incomp(x[0], x[1], x[2]), name='invars_J')([C_bar,H,numTens]) # [?,nSteps,numTens]
    else:
        I = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1]), name='invars_I')([C,H,numTens]) # [?,nSteps,numTens]
        J = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]), name='invars_J')([C,H,numTens]) # [?,nSteps,numTens]
    det_C = tf.keras.layers.Lambda(lambda C: CM.invariant_I3(C), name='invar_det_C')(C)
    
    # referential invariants
    if incomp:
        # isochoric referential invariants
        I_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]) , name='invars_I_ref')([C_bar_ref,H,numTens]) # [?,nSteps,numTens]
        J_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_J_incomp(x[0], x[1], x[2]) , name='invars_J_ref')([C_bar_ref,H,numTens]) # [?,nSteps,numTens]
    else:
        I_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]) , name='invars_I_ref')([C_ref,H,numTens]) # [?,nSteps,numTens]
        J_ref = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]) , name='invars_J_ref')([C_ref,H,numTens]) # [?,nSteps,numTens]
    det_C_ref = tf.keras.layers.Lambda(lambda C_ref: CM.invariant_I3(C_ref), name='invar_det_C_ref')(C_ref)  # [?,nSteps,1]
        
    # invariants of the strain rate tensor
    if incomp:
        I_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]), name='invars_I_bar_dot')([C_bar_dot,H,numTens]) # [?,nSteps,numTens]
        J_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]), name='invars_J_bar_dot')([C_bar_dot,H,numTens]) # [?,nSteps,numTens]
        det_C_dot = tf.keras.layers.Lambda(lambda C_bar_dot: CM.invariant_I3(C_bar_dot), name='invar_det_C_bar_dot')(C_bar_dot) # [?,nSteps,1]           
    else:
        I_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_I(x[0], x[1], x[2]), name='invars_I_dot')([C_dot,H,numTens]) # [?,nSteps,numTens]
        J_dot     = tf.keras.layers.Lambda(lambda x: CM.invariants_J_comp(x[0], x[1], x[2]), name='invars_J_dot')([C_dot,H,numTens]) # [?,nSteps,numTens]
        det_C_dot = tf.keras.layers.Lambda(lambda C_dot: CM.invariant_I3(C_dot), name='invar_det_C_dot')(C_dot) # [?,nSteps,1]   

    ### Concatenate invariants
    Invars     = tf.keras.layers.concatenate([I, J]        , name='concat_invars') 
    Invars_ref = tf.keras.layers.concatenate([I_ref, J_ref], name='concat_invars_ref')
    Invars_dot = tf.keras.layers.concatenate([I_dot, J_dot, det_C_dot], name='concat_invars_dot')

    ### Hyperelasticity
    Invars_in = tf.keras.Input(shape=(nSteps,numTens*2,), name='Invars_in')
        
    # Loop over all generalized structural tensors
    IJ_split = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens*2, axis=-1), name='split_IJ')(Invars_in)
    Psi_subANNs_list = []
    for ii in range(numTens):
        if numExtra == 0:
            IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii]], name='concat_IJ_'+str(ii))
        else:
            IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii], extra_in], name='concat_IJ_extra_'+str(ii))
        
        ## create model for strain energy sub ANN
        Psi_ann = subANNs.Psi_subANN(IJ_in, layer_size_psi, activations_psi, str(ii+1))
        Psi_subANNs_list.append(Psi_ann)
           
    # sum up the strain energy subANN contributions; instead of summing up the individual strain energy contributions,
    # it is possivle to insert a neural network here and nonlinearly mix the individual strain energy contributions 
    # Psi_total = tf.keras.layers.Add(name='Psi_total')(Psi_subANNs)
    
    if numTens == 1:
        Psi_subANNs = Psi_subANNs_list[0] # quick-fix: otherwise tf.keras throws an error when loading the model and tf.keras.concatenate is called on a list with only one element/strain energy contribution (Issue #127 on Github tf-keras)
    else:
        Psi_subANNs = tf.keras.layers.concatenate(Psi_subANNs_list, axis=-1, name='concat_psi_subAnns') 
    
    if numExtra == 0:
        inputs = Invars_in
    else:
        inputs = [Invars_in, extra_in]
    model_Psi = tf.keras.models.Model(inputs=inputs, outputs=Psi_subANNs, name='model_Psi')
    
    # Evaluate Strain Energy Models
    if numExtra == 0:
        Psi_    = model_Psi(Invars)
        Psi_ref = model_Psi(Invars_ref)
    else: 
        Psi_    = model_Psi([Invars, extra_in])
        Psi_ref = model_Psi([Invars_ref, extra_in])
    
    ### isochoric contribution
    Psi_    = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens, axis=-1), name='split_Psi')(Psi_)
    Psi_ref = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens, axis=-1), name='split_Psi_ref')(Psi_ref)

    ### apply offset for stress-free reference configuration; follows Linden et al., 2023
    dPsidI_ref = [CM.GradientLayer(scale=False, name='dPsidI_ref_{:}'.format(ii+1))(psi, I_ref)[:,:,ii:ii+1] for ii, psi in enumerate(Psi_ref)] # [numTens * (?, nSteps, 1)]
    dPsidJ_ref = [CM.GradientLayer(scale=False, name='dPsidJ_ref_{:}'.format(ii+1))(psi, J_ref)[:,:,ii:ii+1] for ii, psi in enumerate(Psi_ref)] # [numTens * (?, nSteps, 1)]
    delta = [tf.keras.layers.Subtract(name='delta_{:}'.format(ii))([dpdi, dpdj]) for ii, (dpdi, dpdj) in enumerate(zip(dPsidI_ref, dPsidJ_ref),1) ] # [numTens * (?, nSteps, 1)]
    alpha = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(-x), name='alpha_{:}'.format(ii))(d) for ii, d in enumerate(delta,1)] # [numTens * (?, nSteps, 1)]
    beta  = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(x),  name='beta_{:}'.format(ii))(d)  for ii, d in enumerate(delta,1)] # [numTens * (?, nSteps, 1)]
    
    Psi_sigma = [CM.PsiSigmaLayer(name='Psi_sigma_{:}'.format(ii+1))(a, b, I[:,:,ii:ii+1], J[:,:,ii:ii+1]) for ii, (a,b) in enumerate(zip(alpha,beta))] # [numTens * (?, nSteps, 1)]
    S_sigma = [CM.GradientLayer(scale=True, name='S_sigma_{:}'.format(ii))(psi_s, C) for ii, psi_s in enumerate(Psi_sigma,1)] # [numTens * (?, nSteps, 3, 3)]

    ### stress (with custom layer)
    S_e_ =    [CM.GradientLayer(scale=True, name='dPsidC_{:}'.format(ii))(psi, C) for ii, psi in enumerate(Psi_,1)] #  [numTens * (?, nSteps, 3, 3)]
    
    ### for debugging
    # dIdC = tf.keras.layers.Lambda(lambda x: tf.gradients(x[0], x[1], unconnected_gradients='zero')[0])([I, C])
    # dJdC = tf.keras.layers.Lambda(lambda x: tf.gradients(x[0], x[1], unconnected_gradients='zero')[0])([J, C])    
    # -----
    # dPsidI = [tf.keras.layers.Lambda(lambda x: tf.gradients(x[0], x[1]), name='dPsidI_{:}'.format(ii))([psi, I]) for ii, psi in enumerate(Psi_,1)]
    # dPsidJ = [tf.keras.layers.Lambda(lambda x: tf.gradients(x[0], x[1]), name='dPsidJ_{:}'.format(ii))([psi, J]) for ii, psi in enumerate(Psi_,1)]
    # -----
    # S_e_bar_ =[tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]), name='dPsidCbar_{:}'.format(ii))([psi, C_bar]) for ii, psi in enumerate(Psi,1)]
    # dS_e_bar_dCbar = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([s[0,:,0,0], C_bar]) for s in S_e_bar_]
    # dS_e_dC = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([s[0,:,0,0], C]) for s in S_e_]
    # -----

    ### apply offset for energy-free reference configuration
    if numTens == 1: # TensorFlow requires special treatment if numTens==1, otherwise problems during prediction
        Psi_    = Psi_[0]
        Psi_ref = Psi_ref[0]
        Psi_sigma = Psi_sigma[0]
    else:
        Psi_      = tf.keras.layers.concatenate(Psi_, axis=-1, name='concat_psi') # (?, nSteps, numTens)
        Psi_ref   = tf.keras.layers.concatenate(Psi_ref, axis=-1, name='concat_psi_ref') # (?, nSteps, numTens)  
        Psi_sigma = tf.keras.layers.concatenate(Psi_sigma, axis=-1, name='concat_psi_sigma') # (?, nSteps, numTens)  
        
    Psi = tf.keras.layers.Add(name='Psi')([Psi_, -Psi_ref, Psi_sigma]) # (?, nSteps, numTens)
        
    ### apply offset for stress-free reference configuration
    if numTens == 1:
        S_e_ = S_e_[0]
        S_sigma = S_sigma[0]
    else:
        S_e_    = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_e')(S_e_) # (?, nSteps, numTens, 3, 3)
        S_sigma = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_sigma')(S_sigma) # (?, nSteps, numTens, 3, 3)
    
    S_e = tf.keras.layers.Add(name='S_e')([S_e_, S_sigma])  # (?, nSteps, numTens, 3, 3)
    
    ### for debugging
    # dSdC_1 = tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_e[:,:,0,0,0], C])
    # dSdC_2 = tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_e[:,:,1,0,0], C])
    # dSdC_3 = tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_e[:,:,2,0,0], C])

    # unstack equilibrium elastic stress into its individual components
    if numTens == 1:
        S_e = [S_e]
    else:
        S_e = tf.keras.layers.Lambda(lambda x: tf.unstack(x, numTens, axis=2), name='unstack_S_e')(S_e) # [numTens * (?, nSteps, 3, 3)]

    # apply incompressibility condition for the equilibrium stress
    S_infy_i = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_infy_incomp_{:}'.format(ii))([s_e, C])  for ii, s_e in enumerate(S_e, 1)]# [numTens*(?, nSteps, 3, 3)]
    S_infy   = tf.keras.layers.Add(name='S_infy')(S_infy_i)

    ### for relaxation experiments
    # S_e = tf.ones_like(F) 
    # S_e = tf.ones_like(F) 

    # Time
    time = tf.keras.layers.Input(shape=(nSteps,), name = 'time_input')
            
    ### viscous part
    if visco == True:
           
        S = []
        S_neq = []
        
        Invars_dot_in = tf.keras.Input(shape=(nSteps,numTens*2+1,), name='Invars_dot_in')       
        if uncoupled:
            IJ_dot_split = tf.keras.layers.Lambda(lambda x: tf.split(x, numTens*2+1, axis=-1), name='split_IJ_dot')(Invars_dot_in)
        
        else:
            IJ_in = Invars_in 
            if rateDependent:
                IJ_in = tf.keras.layers.concatenate([IJ_in, Invars_dot_in], name='prony_concat_invars_dot')
            if numExtra != 0:                
                IJ_in = tf.keras.layers.concatenate([IJ_in, extra_in], name='prony_concat_extra')
        
        Taus_list = []
        Psi_a_list = []

        # Loop over each generalized structural tensor   
        for ii in range(numTens):
            if uncoupled:
                IJ_in = tf.keras.layers.concatenate([IJ_split[ii], IJ_split[numTens+ii]], name='prony_concat_IJ_'+str(ii))
                if rateDependent:
                    IJ_in_t = tf.keras.layers.concatenate([IJ_in, IJ_dot_split[ii], IJ_dot_split[numTens+ii], IJ_dot_split[-1]], name='concat_IJ_t_'+str(ii))
                else:
                    IJ_in_t = IJ_in
                if numExtra != 0:
                    IJ_in   = tf.keras.layers.concatenate([IJ_in, extra_in], name='prony_concat_IJ_extra_'+str(ii))
                    IJ_in_t = tf.keras.layers.concatenate([IJ_in_t, extra_in], name='concat_IJ_t_extra_'+str(ii))

            ### deformation (rate) dependent relaxation time
            Tau_subANNs = []
            for jj in range(nMaxwell):
                Tau_ann = subANNs.Tau_subANN(IJ_in_t, layer_size_tau, activations_tau, '{:}_{:}'.format(ii+1,jj+1))
                Tau_subANNs.append(Tau_ann)        
            
            Tau_subANNs = tf.keras.layers.concatenate(Tau_subANNs, axis=-1, name='concat_tau_{:}'.format(ii))
    
            # scale the relaxation times
            Tau_subANNs = layers.ScaleLayer(tau_min=tau[0], tau_max=tau[1], name='scale_layer_{}'.format(ii+1))(Tau_subANNs)
            
            ### potentials Psi_a   
            Psi_a_subANNs = []
            for jj in range(nMaxwell):
                ### create model for strain energy sub ANN
                Psi_a_ann = subANNs.Psi_a_subANN(IJ_in, layer_size_psi_a, activations_psi_a, '{:}_{:}'.format(ii+1,jj+1))
                Psi_a_subANNs.append(Psi_a_ann)
                
            if nMaxwell == 1:
                Psi_a_subANNs = Psi_a_subANNs[0] # quick-fix: otherwise tf.keras throws an error when loading the model and tf.keras.concatenate is called on a list with only one element/strain energy contribution (Issue #127 on Github tf-keras)
            else:
                Psi_a_subANNs = tf.keras.layers.concatenate(Psi_a_subANNs, axis=-1, name='concat_Psi_a_subAnns_{:}'.format(ii))

            # sparsity regularization
            if lambda_maxwell == 0.0:
                trainable = False
            else:
                trainable = True
            L1_Conv1D = tf.keras.layers.DepthwiseConv1D(
                                    kernel_size=1,
                                    strides=1,
                                    depth_multiplier=1,
                                    activation=None,
                                    use_bias=False,
                                    depthwise_initializer=tf.keras.initializers.ones,
                                    depthwise_regularizer=utils.SparsityRegularizer(l1=lambda_maxwell), #tf.keras.regularizers.L1(l1=lambda_maxwell),
                                    depthwise_constraint=tf.keras.constraints.NonNeg(),
                                    name='l1_layer_{:}'.format(ii),
                                    trainable=trainable,
                                )
            
            Psi_a_subANNs = L1_Conv1D(Psi_a_subANNs) # (?, nSteps, nMaxwell)
            
            inputs_p = [Invars_in]
            inputs_t = [Invars_in]
            if rateDependent:
                inputs_t.append(Invars_dot_in)
            if numExtra != 0:
                inputs_p.append(extra_in)
                inputs_t.append(extra_in)

            model_Psi_a_ = tf.keras.models.Model(inputs=inputs_p, outputs=Psi_a_subANNs, name='model_psi_a_{:}'.format(ii))
            model_tau   = tf.keras.models.Model(inputs=inputs_t, outputs=Tau_subANNs, name='model_tau_{:}'.format(ii))

            # Evaluate Strain Energy Models
            if numExtra == 0:
                Psi_a_    = model_Psi_a_(Invars)
                Psi_a_ref = model_Psi_a_(Invars_ref)    
            else: 
                Psi_a_    = model_Psi_a_([Invars, extra_in])
                Psi_a_ref = model_Psi_a_([Invars_ref, extra_in])
                
            ### isochoric contribution
            Psi_a_      = tf.keras.layers.Lambda(lambda x: tf.split(x, nMaxwell, axis=-1), name='split_Psi_a_N_{:}'.format(ii))(Psi_a_)
            Psi_a_ref = tf.keras.layers.Lambda(lambda x: tf.split(x, nMaxwell, axis=-1), name='split_Psi_a_ref_{:}'.format(ii))(Psi_a_ref)
               
            ### apply offset for stress-free reference configuration; follows Linden et al., 2023
            dPsi_a_dI_ref = [CM.GradientLayer(scale=False, name='dPsi_a_dI_ref_{:}_{:}'.format(ii+1,jj+1))(psi, I_ref)[:,:,ii:ii+1] for jj, psi in enumerate(Psi_a_ref)] # [nMaxwell * (?, nSteps, 1)]
            dPsi_a_dJ_ref = [CM.GradientLayer(scale=False, name='dPsi_a_dJ_ref_{:}_{:}'.format(ii+1,jj+1))(psi, J_ref)[:,:,ii:ii+1] for jj, psi in enumerate(Psi_a_ref)] # [nMaxwell * (?, nSteps, 1)]
            
            delta_a = [tf.keras.layers.Subtract(name='delta_a_{:}_{:}'.format(ii+1,jj))([dpdi, dpdj]) for jj, (dpdi, dpdj) in enumerate(zip(dPsi_a_dI_ref, dPsi_a_dJ_ref),1) ] # [nMaxwell * (?, nSteps, 1)]
            alpha_a = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(-x), name='alpha_a_{:}_{:}'.format(ii+1,jj))(d) for jj, d in enumerate(delta_a,1)] # [nMaxwell * (?, nSteps, 1)]
            beta_a  = [tf.keras.layers.Lambda(lambda x: tf.nn.relu(x),  name='beta_a_{:}_{:}'.format(ii+1,jj))(d)  for jj, d in enumerate(delta_a,1)] # [nMaxwell * (?, nSteps, 1)]
            
            Psi_a_sigma = [CM.PsiSigmaLayer(name='Psi_sigma_{:}_{:}'.format(ii+1,jj+1))(a, b, I[:,:,ii:ii+1], J[:,:,ii:ii+1]) for jj, (a,b) in enumerate(zip(alpha_a,beta_a))] # [nMaxwell * (?, nSteps, 1)]
            S_a_sigma = [CM.GradientLayer(scale=True, name='S_sigma_{:}_{:}'.format(ii+1,jj))(psi_s, C_bar) for jj, psi_s in enumerate(Psi_a_sigma,1)] # [nMaxwell * (?, nSteps, 3, 3)]

            ### stress (with custom layer)
            S_a_ =    [CM.GradientLayer(scale=True, name='dPsi_a_dC_{:}_{:}'.format(ii+1,jj))(psi, C_bar) for jj, psi in enumerate(Psi_a_,1)] #  [nMaxwell * (?, nSteps, 3, 3)]
            
            ### for debugging
            # S_a_i_incomp     = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_a_incomp_{:}'.format(ii))([s_e, C])    for ii, s_e   in enumerate(S_a_, 1)]# [nMaxwell*(?, nSteps, 3, 3)]
            # S_a_sigma_i_incomp = [tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_a_sigma_incomp_{:}'.format(ii))([s_sig, C])  for ii, s_sig in enumerate(S_a_sigma, 1)]# [nMaxwell*(?, nSteps, 3, 3)]
               
            ### apply offset for energy-free reference configuration
            if nMaxwell == 1: # TensorFlow requires special treatment if nMaxwell==1, otherwise problems during prediction
                Psi_a_    = Psi_a_[0]
                Psi_a_ref = Psi_a_ref[0]
                Psi_a_sigma = Psi_a_sigma[0]
            else:
                Psi_a_      = tf.keras.layers.concatenate(Psi_a_, axis=-1, name='concat_Psi_a_{:}'.format(ii)) # (?, nSteps, nMaxwell)
                Psi_a_ref   = tf.keras.layers.concatenate(Psi_a_ref, axis=-1, name='concat_Psi_a_ref_{:}'.format(ii)) # (?, nSteps, nMaxwell)  
                Psi_a_sigma = tf.keras.layers.concatenate(Psi_a_sigma, axis=-1, name='concat_Psi_a_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell)  
                
            Psi_a       = tf.keras.layers.Add(name='Psi_a_{:}'.format(ii))([Psi_a_, -Psi_a_ref, Psi_a_sigma]) # (?, nSteps, nMaxwell)
            Psi_a_s     = tf.keras.layers.Lambda(lambda x: tf.split(x, nMaxwell, axis=-1), name='split_Psi_a_{:}'.format(ii))(Psi_a) # [nMaxwell*(?, nSteps, 1)]
            Psi_a_list.append(Psi_a)
            
            # partial derivatives with respect to I
            dPsi_a_dI    = [CM.GradientLayer(scale=False, name='dPsi_a_dI_{:}_{:}'.format(ii+1,jj+1))(psi, I)[:,:,ii:ii+1]  for jj, psi in enumerate(Psi_a_s)] # [nMaxwell * (?, nSteps, 1)]
            ddPsi_a_ddI  = [CM.GradientLayer(scale=False, name='ddPsi_a_ddI_{:}_{:}'.format(ii+1,jj+1))(dpsi_di,I)[:,:,ii:ii+1]  for jj, dpsi_di in enumerate(dPsi_a_dI)] # [nMaxwell * (?, nSteps, 1)]
            ddPsi_a_dIdJ = [CM.GradientLayer(scale=False, name='ddPsi_a_dIdJ_{:}_{:}'.format(ii+1,jj+1))(dpsi_di,J)[:,:,ii:ii+1]  for jj, dpsi_di in enumerate(dPsi_a_dI)] # [nMaxwell * (?, nSteps, 1)]
                        
            # partial derivatives with respect to J
            dPsi_a_dJ    = [CM.GradientLayer(scale=False, name='dPsi_a_dJ_{:}_{:}'.format(ii+1,jj+1))(psi, J)[:,:,ii:ii+1]  for jj, psi in enumerate(Psi_a_s)] # [nMaxwell * (?, nSteps, 1)]
            ddPsi_a_ddJ  = [CM.GradientLayer(scale=False, name='ddPsi_a_ddJ_{:}_{:}'.format(ii+1,jj+1))(dpsi_dj,J)[:,:,ii:ii+1]  for jj, dpsi_dj in enumerate(dPsi_a_dJ)] # [nMaxwell * (?, nSteps, 1)]
            ddPsi_a_dJdI = [CM.GradientLayer(scale=False, name='ddPsi_a_dJdI_{:}_{:}'.format(ii+1,jj+1))(dpsi_dj,I)[:,:,ii:ii+1]  for jj, dpsi_dj in enumerate(dPsi_a_dJ)] # [nMaxwell * (?, nSteps, 1)]

            # concatenate partial derivatives
            dPsi_a_dI_c    = tf.keras.layers.concatenate(dPsi_a_dI,    axis=-1, name='concat_dPsi_a_dI_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell) 
            dPsi_a_dJ_c    = tf.keras.layers.concatenate(dPsi_a_dJ,    axis=-1, name='concat_dPsi_a_dJ_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell) 
            ddPsi_a_ddI_c  = tf.keras.layers.concatenate(ddPsi_a_ddI,  axis=-1, name='concat_ddPsi_a_ddI_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell) 
            ddPsi_a_ddJ_c  = tf.keras.layers.concatenate(ddPsi_a_ddJ,  axis=-1, name='concat_ddPsi_a_ddJ_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell) 
            ddPsi_a_dIdJ_c = tf.keras.layers.concatenate(ddPsi_a_dIdJ, axis=-1, name='concat_ddPsi_a_dIdJ_sigma_{:}'.format(ii)) # (?, nSteps, nMaxwell) 
                        
            ### apply offset for stress-free reference configuration
            if nMaxwell == 1:
                S_a_ = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2, name='expand_dim_S_a_{:}'.format(ii)))(S_a_[0])
                S_a_sigma = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2, name='expand_dim_S_a_{:}'.format(ii)))(S_a_sigma[0])
            else:
                S_a_      = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_a_{:}'.format(ii))(S_a_) # (?, nSteps, nMaxwell, 3, 3)
                S_a_sigma = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=2), name='stack_S_a_sigma_{:}'.format(ii))(S_a_sigma) # (?, nSteps, nMaxwell, 3, 3)    
            
            S_a = tf.keras.layers.Add(name='S_a_{:}'.format(ii))([S_a_, S_a_sigma])  # (?, nSteps, nMaxwell, 3, 3)

            ###
            # Evaluate relaxation time models           
            inputs = [Invars]
            if rateDependent:
                inputs.append(Invars_dot)
            if numExtra != 0:
                inputs.append(extra_in)
            
            Taus = model_tau(inputs)
            Taus_list.append(Taus)
                        
            ### Internal variable

            Q = tf.keras.layers.Lambda(CM.stressUpdate_Liu, arguments={'nMaxwell':nMaxwell, 'nSteps':nSteps}, name='Q_{:}'.format(ii+1))([S_a, time, Taus])
            Q_unstack = tf.keras.layers.Lambda(lambda x: tf.unstack(x, nMaxwell, axis=2), name='unstack_Q_{:}'.format(ii+1))(Q) # [nMaxwell * (?, nSteps, 3, 3)]
            
            ### for debugging
            # dQ_dCbar_1 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([Q_unstack[0][0,ii,0,0], C_bar]) for ii in range(10)]
            # dQ_dCbar_2 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([Q_unstack[1][0,ii,0,0], C_bar]) for ii in range(10)]
            # dQ_dCbar_3 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([Q_unstack[2][0,ii,0,0], C_bar]) for ii in range(10)]
            # ---
            # # tangent of the fictitious stress with respect to C_bar
            # dS_a_dC_bar = tf.keras.layers.Lambda(lambda x: CM.batch_jacobian_fn(x[0], x[1]))([S_a, C_bar])
            # dS_a_dC_bar = tf.keras.layers.Lambda(lambda x: tf.unstack(x, nMaxwell, axis=2), name='unstack_dS_a_dC_bar_{:}'.format(ii+1))(dS_a_dC_bar)
            # ---
            # dS_a_dC_bar = tf.keras.layers.Lambda(lambda x: CM.tangent(x[0], x[1], x[2], x[3], x[4]))([S_a, C_bar, nMaxwell, nSteps, batchSize]) # (?, 300, 3, 3, 3, 3)
            # dS_a_dC_bar = CM.TangentLayer(nMaxwell, nSteps)(S_a, C_bar)
            # dSdC = tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_a[:,:,0,0,0], C_bar])
                
            
            ### fictitious non-equilibirum stress 
            
            # general formula (not sure if this is correct, needs also elasticity tensor, computationally expensive)
            # S_a_neq_bar = [tf.keras.layers.Lambda(lambda x: tf.einsum('ijklmn,ijmn->ijlk', x[0], x[1]), name='S_a_neq_bar_{:}_{:}'.format(ii+1,jj))([dsdc, q]) for jj, (dsdc,q) in enumerate(zip(dS_a_dC_bar, Q_unstack),1)]
                            
            # explicit formula
            inp = [dPsi_a_dJ_c, ddPsi_a_ddI_c, ddPsi_a_ddJ_c, ddPsi_a_dIdJ_c, C_bar, Q, H[:,:,ii,:,:]]
            S_a_neq_bar = tf.keras.layers.Lambda(lambda x: CM.compute_S_a_neq_bar(x[0], x[1], x[2], x[3], x[4], x[5], x[6]))(inp)
            # # for debugging; unstack to check individual contributions and compre with UMAT
            # S_a_neq_bar_unstack = tf.keras.layers.Lambda(lambda x: tf.unstack(x, nMaxwell, axis=2), name='unstack_S_a_neq_bar_{:}'.format(ii+1))(S_a_neq_bar) # [nMaxwell * (?, nSteps, 3, 3)]
            # dS_a_neq_bar_1 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_a_neq_bar_unstack[0][0,ii,0,0], C_bar]) for ii in range(10)]
            # dS_a_neq_bar_2 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_a_neq_bar_unstack[1][0,ii,0,0], C_bar]) for ii in range(10)]
            # dS_a_neq_bar_3 = [tf.keras.layers.Lambda(lambda x: CM.grad(x[0], x[1]))([S_a_neq_bar_unstack[2][0,ii,0,0], C_bar]) for ii in range(10)]
            
            S_a_neq_bar = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))(S_a_neq_bar)    # accumulate the fictitious nonequilibrium stresses of all Maxwell elements 

            # actual nonequilibrium stress corresponding to a generalized structural tensor
            S_a_neq_dev = tf.keras.layers.Lambda(lambda x: CM.deviatoric_projection(x[0], x[1]), name='S_a_neq_dev_{:}'.format(ii+1))([S_a_neq_bar, C])
            
            # apply incompressibility condition for the nonequilibirum stress
            S_a_neq = tf.keras.layers.Lambda(lambda x: CM.ten2_S_incomp(x[0],x[1]), name='S_a_neq_incomp_{:}'.format(ii+1))([S_a_neq_dev, C])  # (?, nSteps, 3, 3)
        
            S_neq.append(S_a_neq)
                    
        S_i = [ tf.keras.layers.Add(name='S_{:}'.format(nn))([s_infy, s_neq]) for nn, (s_infy, s_neq) in enumerate(zip(S_infy_i, S_neq),1) ]
        
        
    if not visco:
        S_i = S_infy_i
    
    if numTens == 1:
        S = S_i[0]
    else:
        S = tf.keras.layers.add(S_i, name='add_S_i')
        
  
    P, _, _ = CM.stressTensors(S, F)

    ### for relaxation experiments
    # shaper = nSteps*tf.constant([0,1,0,0]) + tf.constant([1,0,3,3])
    # P = tf.keras.layers.Lambda(lambda P: tf.divide(P, tf.tile(tf.expand_dims(P[:,0,0:1,0:1],1), shaper)), name='P_norm')(P)
      

    # Create and return models
    inputs = [F, time]  
    if rateDependent:
        inputs.append(F_dot)
    if numExtraStruc != 0:
        inputs.append(extra_struc_in)
    if numExtra != 0:
        inputs.append(extra_in)

    model_fit  = tf.keras.models.Model(inputs, P) # The fitting model only returning the 1st Piola-Kirchhoff stress

    # for debugging
    model_full =  tf.keras.models.Model(inputs, [Invars, Psi, Psi_,
                                                #  dPsidI, dPsidJ,
                                                 dPsidI_ref, dPsidJ_ref, 
                                                 Psi_sigma, delta,
                                                 alpha, beta, 
                                                 S_e_,
                                                 S_sigma, S_e,
                                                 S_infy_i, S_infy,
                                                 Taus_list, S,
                                                 S_i, S_neq,
                                                 Psi_a_, Psi_a_ref,
                                                 Psi_a_sigma, Psi_a_list,
                                                 dPsi_a_dI_ref, dPsi_a_dJ_ref,
                                                 delta_a, alpha_a,
                                                 beta_a, S_a_sigma,
                                                 S_a_, S_a,
                                                 Q_unstack, P,
                                                # dSdC_1,
                                                #  dS_a_dC_bar,
                                                #  S_a_neq_bar, S_a_neq_bar_ex,
                                                #  S_e_i_incomp, S_sigma_i_incomp,
                                                #  S_a_i_incomp, S_a_sigma_i_incomp,
                                                #  dPsi_a_dI_c, dPsi_a_dJ_c, 
                                                #  ddPsi_a_ddI_c, ddPsi_a_ddJ_c, 
                                                #  ddPsi_a_dIdJ_c, H, dS_a_neq_bar_1, dS_a_neq_bar_2, dS_a_neq_bar_3
                                                 ]) # The full one with all values of interest

    return model_fit, model_full

