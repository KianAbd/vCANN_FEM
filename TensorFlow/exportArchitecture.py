# Standard imports
import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import shutil
from contextlib import redirect_stdout
from matplotlib import cm
import json

from utils import R2
from natsort import natsorted

# map activation name to integer code used in Fortran (tanh=1, sigmoid=2, softplus=3, linear=4, elu=5)
def act_name_to_code(name):
    n = name.lower()
    if n == 'tanh':
        return 1
    elif n == 'sigmoid':
        return 2
    elif n == 'softplus':
        return 3
    elif n == 'linear' or n == 'none':
        return 4
    elif n == 'elu':
        return 5
    elif n == 'squared_softplus':
        return 6
    else:
        raise ValueError(f"Activation function '{name}' not recognized. Must be one of: tanh, sigmoid, softplus, linear, elu, squared_softplus.")


def _format_fortran_value_block(values, indent="      ", width=80):
    """Return a multiline Fortran-formatted list of values with continuation marks."""
    flat = np.asarray(values).flatten()
    if flat.size == 0:
        return indent + "0.0_dp &\n"

    lines = []
    line = indent
    for idx, val in enumerate(flat):
        sval = f"{float(val):.16e}_dp"
        if idx < flat.size - 1:
            sval += ", "
        if len(line) + len(sval) > width:
            lines.append(line + " &\n")
            line = indent + sval
        else:
            line += sval
    lines.append(line + " &\n")
    return "".join(lines)


def format_fortran_reshape(name, array, shape, indent="      "):
    """Format a numpy array as a Fortran reshape assignment."""
    if isinstance(shape, (tuple, list)):
        shape_str = "[" + ", ".join(str(int(dim)) for dim in shape) + "]"
    else:
        shape_str = str(shape)
    body = _format_fortran_value_block(array, indent=indent)
    return f"{name} = reshape([ &\n{body}  ], {shape_str}) \n"


def format_layer_weights(layer, indent="      "):
    """Return Fortran declaration for a Dense layer's kernel."""
    rows = layer.kernel.shape[1]
    cols = layer.kernel.shape[0]
    reshape_str = format_fortran_reshape(f"W_{layer.name}", layer.kernel.numpy(), (cols, rows), indent)
    return f"  real(dp), dimension({rows},{cols}), parameter :: {reshape_str}"


def format_layer_bias(layer):
    """Return Fortran declaration for a Dense layer's bias vector."""
    size = layer.kernel.shape[1]
    prefix = f"  real(dp), dimension({size}), parameter :: b_{layer.name} "
    if layer.use_bias:
        bias_str = ", ".join(f"{float(val):.16e}_dp" for val in layer.bias.numpy())
        return f"{prefix}= [ {bias_str} ]\n\n"
    return f"{prefix}= 0.0_dp \n\n"


def join_fortran_lines(lines, newline=True):
    """Join pre-indented lines with newlines, optionally adding a trailing newline."""
    block = "\n".join(lines)
    return block + ("\n" if newline else "")

def exportArchitecture(model_fit, outputFolder, save_local=True):
    

    ### set pathes and create folders 
    
    # save weights, UMAT, ... in the outputFolder
    if save_local == True:       
        umat_folder = outputFolder + '\\UMAT'
        if not os.path.isdir(umat_folder):
            os.makedirs(umat_folder)
        
    # save weights, UMAT, for Visual Studio inspection/debugging ...
    else:
        umat_folder = "..\\Abaqus\\"

    #
    models = []
    for layer in model_fit.layers:
        if 'model' in layer.name:
            models.append(layer)

    H_layer = model_fit.get_layer('H') # generalized structural tensor layer
    numTens = H_layer.input[-1].numpy() # number of generalized structural tensors
    numDir = H_layer.input[-2].numpy() # number of preferred material directions
    
    # additional inputs for the fiber orientations and weights of the generalized structural tensors
    if any(layer.name == 'extra_struc_input' for layer in model_fit.layers):
        extra_struc_in = model_fit.get_layer('extra_struc_input')
        numExtraStruc = extra_struc_in.input.shape.as_list()[-1]
    else:
        numExtraStruc = 0

    # additional inputs for the fiber orientations and weights of the generalized structural tensors
    if any(layer.name == 'extra_input' for layer in model_fit.layers):
        extra_in = model_fit.get_layer('extra_input')
        numFeatures = extra_in.input.shape.as_list()[-1]
    else:
        numFeatures = 0

    #
    ### Equilibrium free energy
    #
    model_Psi = model_fit.get_layer('model_Psi')
    psi_layers = [l for l in model_Psi.layers if 'Psi_1' in l.name]
    
    numLayersPsi = len(psi_layers)
    
    # Network parameters (assumes that all equilibrium free energy subANNs have the same architecture)
    layer_sizes = []
    activations = []
    psi_layers = [l for l in model_Psi.layers if 'Psi_1' in l.name]
    for ii, l in enumerate(psi_layers,1):
        layer_sizes.append(l.kernel.shape)
        activations.append(l.activation.__name__)

    layer_sizes = np.array(layer_sizes) # (numLayersPsi, (numInputs, numOutputs))
    numInputs = layer_sizes[:,0] # (numLayersPsi,)
    numOutputs = layer_sizes[:,1] # (numLayersPsi,)
    in_str = ", ".join(f"{int(x)}" for x in numInputs.flatten())
    out_str = ", ".join(f"{int(x)}" for x in numOutputs.flatten())

    activations = np.array(activations) # (numLayersPsi,)
    act_codes = np.vectorize(act_name_to_code)(activations)
    act_str = ", ".join(f"{int(x)}" for x in act_codes.flatten())

    max_numInputs = np.max(numInputs)
    max_numOutputs = np.max(numOutputs)

    # Compute alpha and beta (for stress offset in the reference configuration)
    F_in = model_fit.get_layer('F_input').input
    nSteps = F_in.shape[1]

    alpha_layers = [l.output for l in model_fit.layers if l.name.startswith('alpha_') and not l.name.startswith('alpha_a_')]
    beta_layers =  [l.output for l in model_fit.layers if l.name.startswith('beta_')  and not l.name.startswith('beta_a_')]

    alphas_concat = tf.keras.layers.Concatenate()(alpha_layers) # (?, 2*numTens)
    betas_concat = tf.keras.layers.Concatenate()(beta_layers) # (?, 2*numTens)

    inputs = [F_in]
    if numFeatures > 0:
        inputs.append(extra_in.input)

    model_alpha = tf.keras.models.Model(inputs=inputs, outputs=alphas_concat) # (?, 2*numTens)
    model_beta = tf.keras.models.Model(inputs=inputs, outputs=betas_concat) # (?, 2*numTens)

    F_ref = np.eye(3)
    F_ref = np.expand_dims(np.expand_dims(F_ref, 0), 0)
    F_ref = np.tile(F_ref, (1, nSteps, 1, 1)) # (1, nSteps, 3, 3)

    inputs = [F_ref]

    if numFeatures > 0:
        features = np.ones((1, nSteps, numFeatures), dtype=np.float64)
        inputs.append(features)
    alphas = model_alpha.predict(inputs)[0,0] # (nSteps, 2*numTens)
    betas = model_beta.predict(inputs)[0,0] # (nSteps, 2*numTens)

    alpha_str = 'alphas = 0.0_dp ! initialized by init_offset()\n' # [' + ", ".join(f"{float(val):.16e}_dp" for val in alphas.flatten()) + ']' # is calculated in the UMAT now
    beta_str = 'betas = 0.0_dp \n\n' # [' + ", ".join(f"{float(val):.16e}_dp" for val in betas.flatten()) + ']' # is calculated in the UMAT now

    #
    ### Relaxation times and non-equilibrium free energy
    #
    
    numMaxwell = model_fit.get_layer('model_tau_0').outputs[0].shape[-1]

    # number of layers of each tau and psi_a subnetwork
    model = model_fit.get_layer('model_tau_0')
    layers = [l for l in model.layers if 'Tau_1' in l.name]
    layers = natsorted(layers, key=lambda x: x.name)
    numLayersTau = len(layers)//numMaxwell

    model = model_fit.get_layer('model_psi_a_0')
    layers = [l for l in model.layers if 'Psi_a_1' in l.name]
    layers = natsorted(layers, key=lambda x: x.name)
    numLayersPsi_a = len(layers)//numMaxwell  


    # Network parameters (assumes that all equilibrium free energy subANNs have the same architecture)
    layer_sizes = []
    activations = []
    prefixes = ['Tau', 'Psi_a']
    for p in prefixes:
        shapes = []
        acts = []
        model = model_fit.get_layer('model_{}_0'.format(p.lower()))
        layers = [l for l in model.layers if '{}_1_1'.format(p) in l.name]
        layers = natsorted(layers, key=lambda x: x.name)
        for ii, l in enumerate(layers,1):
            shapes.append(l.kernel.shape)
            acts.append(l.activation.__name__)
        layer_sizes.append(shapes)
        activations.append(acts)

    layer_sizes_tau = np.array(layer_sizes[0]) # (numLayersTau, (numInputs, numOutputs))
    numInputsTau = layer_sizes_tau[:,0] # (numLayersTau,)
    numOutputsTau = layer_sizes_tau[:,1] # (numLayersTau,)
    in_str_tau = ", ".join(f"{int(x)}" for x in numInputsTau.flatten())
    out_str_tau = ", ".join(f"{int(x)}" for x in numOutputsTau.flatten())

    activationsTau = np.array(activations[0]) # (numLayersTau,)
    act_codes_tau = np.vectorize(act_name_to_code)(activationsTau)
    act_str_tau = ", ".join(f"{int(x)}" for x in act_codes_tau.flatten())

    max_numInputs_tau = np.max(numInputsTau) # (numTens,) 
    max_numOutputs_tau = np.max(numOutputsTau) # (numTens,)

    layer_sizes_psi_a = np.array(layer_sizes[1]) # (numLayersPsi_a, (numInputs, numOutputs))
    numInputsPsi_a = layer_sizes_psi_a[:,0] # (numLayersPsi_a,)
    numOutputsPsi_a = layer_sizes_psi_a[:,1] # (numLayersPsi_a,)
    in_str_psi_a = ", ".join(f"{int(x)}" for x in numInputsPsi_a.flatten())
    out_str_psi_a = ", ".join(f"{int(x)}" for x in numOutputsPsi_a.flatten())

    activationsPsi_a = np.array(activations[1]) # (numLayersPsi_a,)
    act_codes_psi_a = np.vectorize(act_name_to_code)(activationsPsi_a)
    act_str_psi_a = ", ".join(f"{int(x)}" for x in act_codes_psi_a.flatten())

    max_numInputs_psi_a = np.max(numInputsPsi_a)
    max_numOutputs_psi_a = np.max(numOutputsPsi_a)

    scales = []
    L1 = []
    for tt in range(1,numTens+1):
        model_tau = model_fit.get_layer('model_tau_{}'.format(tt-1))
        scale_layer = model_tau.get_layer('scale_layer_{}'.format(tt))
        tau_min = scale_layer.tau_min
        tau_max = scale_layer.tau_max
        scale = np.logspace(tau_min, tau_max, num=numMaxwell, endpoint=True, base=10.0, dtype=np.float64, axis=0)
        scales.append(scale)

        model_psi_a = model_fit.get_layer('model_psi_a_{}'.format(tt-1))
        reg_layer = model_psi_a.get_layer('l1_layer_{}'.format(tt-1))
        kernel = reg_layer.depthwise_kernel.numpy().reshape(-1)
        L1.append(kernel)

    scales = np.array(scales) # (numTens, numMaxwell)
    text_scale = format_fortran_reshape('scale_tau', scales, (numMaxwell, numTens)) + '\n'

    L1 = np.array(L1) # (numTens, numMaxwell)
    text_L1 = format_fortran_reshape('L1', L1, (numMaxwell, numTens)) + '\n'

    
    
    ### Compute alpha and beta (for stress offset in the reference configuration)
    inputs = [F_in]
    if numFeatures > 0:
        inputs.append(extra_in.input)
    
    alphas_a = []
    betas_a = []

    for tt in range(1,numTens+1):

        alpha_a_layers = [l.output for l in model_fit.layers if l.name.startswith(f'alpha_a_{tt}')]
        beta_a_layers = [l.output for l in model_fit.layers if l.name.startswith(f'beta_a_{tt}')]

        alphas_a_concat = tf.keras.layers.Concatenate()(alpha_a_layers)
        betas_a_concat = tf.keras.layers.Concatenate()(beta_a_layers)

        alphas_a.append(alphas_a_concat[0,0]) 
        betas_a.append(betas_a_concat[0,0])

    alphas_a = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=0))(alphas_a)
    betas_a = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=0))(betas_a) 

    model_alpha = tf.keras.models.Model(inputs=inputs, outputs=alphas_a)
    model_beta = tf.keras.models.Model(inputs=inputs, outputs=betas_a)

    F_ref = np.eye(3)
    F_ref = np.expand_dims(np.expand_dims(F_ref, 0), 0)
    F_ref = np.tile(F_ref, (1, nSteps, 1, 1)) # (1, nSteps, 3, 3)

    inputs = [F_ref]
    if numFeatures > 0:
        features = np.ones((1, nSteps, numFeatures), dtype=np.float64)
        inputs.append(features)

    alphas = model_alpha.predict(inputs)
    betas = model_beta.predict(inputs)

    alphas = np.array(alphas) # (numTens, numMaxwell)
    betas = np.array(betas) # (numTens, numMaxwell)

    
    alpha_a_str = 'alphas_a = 0.0_dp ! initialized by init_offset() \n' #  array_to_fortran_reshape(alphas, "alphas_a", f"[{numMaxwell}, {numTens}]") # is calculated in the UMAT now
    beta_a_str = 'betas_a = 0.0_dp \n\n' # array_to_fortran_reshape(betas, "betas_a", f"[{numMaxwell}, {numTens}]") # is calculated in the UMAT now

    #
    ###
    #

    if any(layer.name == 'Invars_dot_in' for layer in model_fit.get_layer('model_tau_0').layers):
        rateDependent = '.TRUE.'
    else:
        rateDependent = '.FALSE.'

    #
    ###
    #

    text_0 =  '!dec$ freeform \n\n'

    text_0 += "include 'precision.f90' \n"
    text_0 += "include 'umat_statev_utils.f90' \n"
    text_0 += "include 'FFNN_derivatives.f90' \n\n"

    text_0 += 'module parameters \n\n'
    text_0 += '  use precision \n'
    text_0 += '  implicit none \n'
    text_0 += '  save \n\n'
   
    text_0 += f'  integer, parameter :: numTens = {numTens} ! number of generalized structural tensors \n'  
    text_0 += f'  integer, parameter :: numDir = {numDir} ! number of preferred material directions \n'
    text_0 += f'  integer, parameter :: numMaxwell = {numMaxwell} ! number of Maxwell elements per generalized structural tensor \n'
    text_0 += f'  integer, parameter :: numFeatures = {numFeatures} ! number of elements of the feature vector \n'
    text_0 += f'  logical, parameter :: rateDependent = {rateDependent} ! flag for rate dependency of relaxation times \n\n'

    text_0 += '  real(dp) :: zero, one, two, three, four, six \n'
    text_0 += '  parameter(zero=0.0_dp, one=1.0_dp, two=2.0_dp, three=3.0_dp, four=4.0_dp, six=6.0_dp) \n\n'

    text_0 += '  ! Identity matrix \n'
    text_0 += '  real(dp), dimension(3,3), parameter :: eye &  \n'
    text_0 += '      = reshape((/one,zero,zero,zero,one,zero,zero,zero,one/),(/3,3/))  \n\n'

    text_0 += '  ! Generalized invariants in the reference configuration \n'
    text_0 += '  real(dp), dimension(2*numTens), parameter :: invarsRef = one  \n'
    text_0 += '  real(dp), dimension(2*numTens+1), parameter :: invarsDotRef = zero  \n\n'

    text_0 += '  real(dp), parameter, dimension(numFeatures) :: features = 1.0_dp  ! feature vector \n'  
    text_0 += '  real(dp), parameter, dimension(1,1) :: extra_struc =  1.0_dp  ! dummy input \n\n'    
 
    text_0 += '  ! UMAT flags \n'
    text_0 += '  logical, parameter :: use_numerical_tangent = .FALSE. ! flag for choosing numerical tangent \n'
    text_0 += '  real(dp), parameter :: epsilon = 2.e-8_dp ! perturbation size for numerical tangent\n\n'

    text_0 += '  logical, parameter :: use_shell = .FALSE. ! flag if shell or membrane elements are used \n'
    text_0 += '  logical, parameter :: local_csys = .FALSE. ! flag if a local coordinate system is used \n\n'

    #
    ### Add the weight loading and variable initialization
    #

    if numDir != 0 and numExtraStruc != 0:

        text_4 = '\n\n'
        text_4 += '! ================================================================ \n'
        text_4 += '! Generalized structural tensors (preferred material directions and weights) \n'
        text_4 += '! ================================================================ \n\n'
        
        num = [numDir, numTens]
        prefixes = ['dir', 'w']
        title = ['Direction l', 'Scalar weighting factors w']
        for p, n, t in zip(prefixes, num, title):
            model = model_fit.get_layer('model_{}'.format(p))
            for kk in range(1,n+1):
                text_4 += '\n! {}_{} \n'.format(t, kk)
                layers = [l for l in model.layers if '{}_{}'.format(p, kk) in l.name]
                numLayers = len(layers)
                for ii, l in enumerate(layers,1):
                    text_4 += '! Layer {} \n'.format(((ii-1)%numLayers+1))
                    kernel = l.kernel.numpy().T
                    kernel.tofile(save_folder + '\\W_{}.bin'.format(l.name))
                    text_4 += "open(unit=10, file=" + data_root_folder + "\\W_{}.bin', access='stream', status='old', action='read') \n".format(l.name)
                    text_4 += 'read(10) W_{} \n'.format(l.name)
                    text_4 += 'close(10) \n'
                    if l.use_bias:
                        l.bias.numpy().tofile(save_folder + '\\b_{}.bin'.format(l.name))                    
                        text_4 += "open(unit=10, file=" + data_root_folder + "\\b_{}.bin', access='stream', status='old', action='read') \n".format(l.name)
                        text_4 += 'read(10) b_{} \n'.format(l.name)
                        text_4 += 'close(10) \n'
                        
    text_4 = '  ! ================================================================ \n'
    text_4 += '  ! Equilibrium free energy \n'
    text_4 += '  ! ================================================================ \n'
    text_4 += '  real(dp), parameter :: kappa= zero ! volumetric contribution - penalty parameter (only for nearly-incompressible material) \n\n'
    text_4 += f'  integer, parameter :: L_eq = {len(psi_layers)}\n'
    text_4 += f'  integer, parameter :: max_in={max_numInputs}, max_out={max_numOutputs}\n'
    text_4 += f'  integer, dimension(L_eq), parameter :: nin = [ {in_str} ], nout = [ {out_str} ], acts = [ {act_str} ]\n'
    text_4 += '  real(dp), dimension(max_out, max_in, L_eq, numTens) :: weights = 0.0_dp\n'
    text_4 += '  real(dp), dimension(max_out, L_eq, numTens) :: biases = 0.0_dp\n\n'

    text_4 += '  ! Stress offset equilibrium free energy \n'
    text_4 += '  real(dp), dimension(numTens) :: ' + alpha_str
    text_4 += '  real(dp), dimension(numTens) :: ' + beta_str

    # Variables for strain energy density functions
    text_1 = 'module vCANNs \n\n'
    text_1 += '  use precision \n'
    text_1 += '  use parameters \n'
    text_1 += '  implicit none \n\n'
    text_1 += '  private\n'
    text_1 += '  public :: vCANN_equi, vCANN_non_equi, init_offset\n\n'
    text_1 += '  contains \n\n'
    text_1 += '  ! ================================================================ \n'
    text_1 += '  ! Equilibrium free energy \n'
    text_1 += '  ! ================================================================ \n'
    text_1 += '  subroutine vCANN_equi(invars, J_Psi, H_Psi)\n\n'

    text_1 += '    use precision\n'
    text_1 += '    use deriv_recursive\n'
    text_1 += '    use parameters\n'
    text_1 += '    implicit none \n\n'

    text_1 += '    ! Network inputs\n'
    text_1 += '    real(dp), dimension(2*numTens), intent(in) :: invars\n'
    text_1 += '    real(dp), dimension(2+numFeatures) :: inputs\n\n'

    text_1 += '    ! Network outputs\n'
    text_1 += '    real(dp), allocatable, intent(out) :: J_Psi(:,:,:), H_Psi(:,:,:,:)\n'
    text_1 += '    real(dp), allocatable :: Psi(:), J(:,:), H(:,:,:), T(:,:,:,:)\n'
    text_1 += '    logical, parameter :: want_hessian=.True., want_third=.False.\n'
    text_1 += '    integer :: ii\n\n'
    
    text_init  = '    allocate(J_Psi(nout(L_eq),nin(1),numTens)); J_Psi = 0.0_dp\n'
    text_init += '    allocate(H_Psi(nout(L_eq),nin(1),nin(1),numTens)); H_Psi = 0.0_dp\n\n'

    text_pad = ''
    text_1_1 = ''
    for tt in range(1,numTens+1):
        text_1_1 += '\n  ! Equilibrium free energy contribution {} \n'.format(tt)
        text_1_1 += '  ! ================================================================ \n'
        
        psi_layers = [l for l in model_Psi.layers if f'Psi_{tt}' in l.name]
        for ii, l in enumerate(psi_layers,1):
            text_1_1 += '  ! Layer {} \n'.format(ii)
            
            if f'Psi_{tt}' in l.name:
                text_1_1 += format_layer_weights(l)
                text_pad += '    weights(1:{}, 1:{}, {}, {}) = W_{} \n'.format(l.kernel.shape[1], l.kernel.shape[0], ii, tt, l.name)

                text_1_1 += format_layer_bias(l)
                text_pad += '    biases(1:{}, {}, {}) = b_{} \n\n'.format(l.kernel.shape[1], ii, tt, l.name)

    text_init_psi_eq = 'contains \n\n'
    text_init_psi_eq += '  subroutine init_weights_eq()\n'
    text_init_psi_eq += '  ! Initialize weights before analysis starts \n'
    text_init_psi_eq += '    implicit none \n'
    text_init_psi_eq += '    save \n\n'
    text_init_psi_eq += text_pad
    text_init_psi_eq += '  end subroutine init_weights_eq \n\n'

    text_eval =  '    ! ================================================================ \n'
    text_eval += '    ! Evaluate ANN and compute derivatives \n'
    text_eval += '    ! ================================================================ \n'
    text_eval += '    do ii = 1, numTens\n\n'

    text_eval += '        if (numFeatures .GT. 0) then \n'
    text_eval += '            inputs(1:2) = invars(2*ii-1:2*ii)\n'
    text_eval += '            inputs(3:3+numFeatures-1) = features\n'
    text_eval += '        else \n'
    text_eval += '            inputs =  invars(2*ii-1:2*ii)\n'
    text_eval += '        end if \n\n'
    
    text_eval += '        call derivatives_output(weights(:,:,:,ii), biases(:,:,ii), acts, inputs,   &\n'
    text_eval += '                                nout, nin, L_eq, want_hessian, want_third, &\n'
    text_eval += '                                J, H, T, Psi)\n\n'
    
    text_eval += '        ! Correct for stress-free reference configuration \n'
    text_eval += '        J(1,1) = J(1,1) + alphas(ii) \n'
    text_eval += '        J(1,2) = J(1,2) + betas(ii) \n\n'
    
    text_eval += '        ! Collect outputs \n'
    text_eval += '        J_Psi(:,:,ii) = J \n'
    text_eval += '        H_Psi(:,:,:,ii) = H \n\n'

    text_eval += '    end do\n\n'

    text_eval += '  end subroutine vCANN_equi \n\n'

    text_1 = text_1 + text_init + text_eval

    text_4 += text_1_1

    
    #
    ### Subroutine Prony Parameters
    #

    text_4 += '  ! ================================================================ \n'
    text_4 += '  ! Non-equilibrium free energy \n'
    text_4 += '  ! ================================================================ \n'

    text_4 += f'  integer, parameter :: L_t = {numLayersTau}\n'
    text_4 += f'  integer, parameter :: L_neq = {numLayersPsi_a}\n\n'

    text_4 += f'  integer, parameter :: max_in_tau = {max_numInputs_tau}, max_out_tau = {max_numOutputs_tau}\n\n'
    text_4 += f'  integer, parameter :: max_in_psi_a = {max_numInputs_psi_a}, max_out_psi_a = {max_numOutputs_psi_a}\n'

    text_4 += f'  integer, dimension(L_t), parameter :: nin_t = [ {in_str_tau} ], nout_t = [ {out_str_tau} ], acts_t = [ {act_str_tau} ]\n'
    text_4 += f'  integer, dimension(L_neq), parameter :: nin_p = [ {in_str_psi_a} ], nout_p = [ {out_str_psi_a} ], acts_p = [ {act_str_psi_a} ]\n\n'

    text_4 += '  ! Network weights and biases\n'
    text_4 += '  real(dp), dimension(max_out_tau, max_in_tau, L_t, numMaxwell, numTens) :: weights_Tau = 0.0_dp\n'
    text_4 += '  real(dp), dimension(max_out_tau, L_t, numMaxwell, numTens) :: biases_Tau = 0.0_dp\n\n'

    text_4 += '  real(dp), dimension(max_out_psi_a, max_in_psi_a, L_neq, numMaxwell, numTens) :: weights_Psi_a = 0.0_dp\n'
    text_4 += '  real(dp), dimension(max_out_psi_a, L_neq, numMaxwell, numTens) :: biases_Psi_a = 0.0_dp\n\n'

    text_4 += '  ! Stress offset non-equilibrium free energy \n'
    text_4 += '  real(dp), dimension(numMaxwell,numTens) :: ' + alpha_a_str
    text_4 += '  real(dp), dimension(numMaxwell,numTens) :: ' + beta_a_str

    text_4 += '  ! Time scaling \n'
    text_4 += '  real(dp), dimension(numMaxwell,numTens), parameter :: ' + text_scale

    text_4 += '  ! L1 sparsity \n'
    text_4 += '  real(dp), dimension(numMaxwell,numTens), parameter :: ' + text_L1

    text_2 =  '  ! ================================================================ \n'
    text_2 += '  ! Non-equilibrium free energy and relaxation time\n'
    text_2 += '  ! ================================================================ \n'
    text_2 += '  subroutine vCANN_non_equi(invars, invars_dot, Tau, J_Tau, J_Psi_a, H_Psi_a, T_Psi_a)\n\n'

    text_2 += '    use precision\n'
    text_2 += '    use deriv_recursive\n'
    text_2 += '    use parameters\n'
    text_2 += '    implicit none \n\n'

    text_2 += '    ! Network inputs\n'
    text_2 += '    real(dp), intent(in) :: invars(:) ! generalized invariants\n'
    text_2 += '    real(dp), intent(in) :: invars_dot(:) ! generalized invariants of the material time derivative of the RCG tensor\n'
    text_2 += '    real(dp), allocatable :: inputs_tau(:)\n'
    text_2 += '    real(dp), allocatable :: inputs_psi_a(:)\n\n'

    text_2 += '    ! Network outputs\n'
    text_2 += '    real(dp), allocatable, intent(out) :: Tau(:,:) \n'
    text_2 += '    real(dp), allocatable, intent(out) :: J_tau(:,:,:,:) \n'
    text_2 += '    real(dp), allocatable, intent(out) :: J_Psi_a(:,:,:,:), H_Psi_a(:,:,:,:,:), T_Psi_a(:,:,:,:,:,:)\n'
    text_2 += '    real(dp), allocatable :: J(:,:), H(:,:,:), T(:,:,:,:)\n'
    text_2 += '    real(dp), allocatable :: y(:)\n\n'
    text_2 += '    logical, parameter :: want_hessian_tau=.False., want_third_tau=.False.\n'
    text_2 += '    logical, parameter :: want_hessian_psi_a=.True., want_third_psi_a=.True.\n'
    text_2 += '    integer :: ii, jj\n\n'

    text_2_2 = ''
    for tt in range(1,numTens+1):
        
        text_2_2 += '\n  ! ================================================================ \n'
        text_2_2 += '  ! Generalized Maxwell model {} \n'.format(tt)

        for p in prefixes:
            model = model_fit.get_layer('model_{}_{}'.format(p.lower(), tt-1))
            layers = [l for l in model.layers if '{}_{}'.format(p, tt) in l.name]
            layers = natsorted(layers, key=lambda x: x.name)
            numLayersPerMaxwell = len(layers)//numMaxwell
                
            for ii, l in enumerate(layers,1):
                if (ii-1) % numLayersPerMaxwell == 0:
                    maxidx = (ii-1)//numLayersPerMaxwell+1
                    if p == '    Tau':
                        text_2_2 += f'  ! Relaxation time tau_{maxidx} \n'.format()
                    else:
                        text_2_2 += f'  ! Non-equilibrium free energy Psi_a_{maxidx} \n'

                    text_2_2 += '  ! ================================================================ \n'

                text_2_2 += '  ! Layer {} \n'.format((ii-1)%numLayersPerMaxwell+1)

                text_2_2 += format_layer_weights(l)
                text_2_2 += format_layer_bias(l)


    text_init_collect  = '    allocate(Tau(numMaxwell,numTens)); Tau = 0.0_dp\n'
    text_init_collect += '    allocate(J_Tau(nout_t(L_t),nin_t(1),numMaxwell,numTens)); J_Tau = 0.0_dp\n\n'

    text_init_collect += '    allocate(J_Psi_a(nout_p(L_neq),nin_p(1),numMaxwell,numTens)); J_Psi_a = 0.0_dp\n'
    text_init_collect += '    allocate(H_Psi_a(nout_p(L_neq),nin_p(1),nin_p(1),numMaxwell,numTens)); H_Psi_a = 0.0_dp\n'
    text_init_collect += '    allocate(T_Psi_a(nout_p(L_neq),nin_p(1),nin_p(1),nin_p(1),numMaxwell,numTens)); T_Psi_a = 0.0_dp\n\n'

    text_pad_collect = ''
    for tt in range(1,numTens+1):  

        for p in prefixes:
                
            text_init = ''
            # text_init += f'    ! Generalized Maxwell model {tt} \n\n'
            
            text_pad = '\n    ! ================================================================ \n'
            text_pad += f'    ! Generalized Maxwell model {tt} \n'

            model = model_fit.get_layer('model_{}_{}'.format(p.lower(), tt-1))
            layers = [l for l in model.layers if '{}_{}'.format(p, tt) in l.name]
            layers = natsorted(layers, key=lambda x: x.name)
            numLayersPerMaxwell = len(layers)//numMaxwell

            for ii, l in enumerate(layers,1):
                if (ii-1) % numLayersPerMaxwell == 0:
                    maxidx = (ii-1)//numLayersPerMaxwell+1
                    if p == '    Tau':
                        t = f'\n    ! Relaxation time tau_{maxidx} \n'
                    else:
                        t = f'\n    ! Non-equilibrium free energy Psi_a_{maxidx} \n'
                    t += '    ! ================================================================ \n'                    
                    # text_init += t
                    text_pad += t

                lidx = (ii-1)%numLayersPerMaxwell+1
                text_pad += '    weights_{}(1:{}, 1:{}, {}, {}, {}) = W_{} \n'.format(p, l.kernel.shape[1], l.kernel.shape[0], lidx, maxidx, tt, l.name)
                text_pad += '    biases_{}(1:{}, {}, {}, {}) = b_{} \n'.format(p, l.kernel.shape[1], lidx, maxidx, tt, l.name)

            text_init_collect += text_init
            text_pad_collect += text_pad


    text_init_psi_neq = '  subroutine init_weights_neq()\n'
    text_init_psi_neq += '  ! Initialize weights before analysis starts \n'
    text_init_psi_neq += '    implicit none \n'
    text_init_psi_neq += '    save \n'
    text_init_psi_neq += text_pad_collect
    text_init_psi_neq += '\n  end subroutine init_weights_neq \n\n'
    text_init_psi_neq += 'end module parameters \n'
    text_init_psi_neq +=  '\n! ================================================================ \n\n'
    

    text_eval =  '    ! ================================================================ \n'
    text_eval += '    ! Evaluate ANN and compute derivatives \n'
    text_eval += '    ! ================================================================ \n'
    text_eval += '    if (rateDependent == .TRUE.) then \n'
    text_eval += '      allocate(inputs_tau(5 + numFeatures)); inputs_tau = 0.0_dp\n'
    text_eval += '    else\n'
    text_eval += '      allocate(inputs_tau(2 + numFeatures)); inputs_tau = 0.0_dp\n'
    text_eval += '    end if\n\n'

    text_eval += '    allocate(inputs_psi_a(2 + numFeatures)); inputs_psi_a = 0.0_dp\n\n'

    text_eval += '    do ii = 1, numTens\n\n'

    text_eval += '        if (rateDependent == .TRUE.) then\n'
    text_eval += '          inputs_tau(1:2) = invars(2*ii-1:2*ii)\n'
    text_eval += '          inputs_tau(3:4) = invars_dot(2*ii-1:2*ii)\n'
    text_eval += '          inputs_tau(5) = invars_dot(2*numTens+1)\n'
    text_eval += '          if (numFeatures .gt. 0) then\n' 
    text_eval += '            inputs_tau(6: 6+numFeatures-1) = features\n'
    text_eval += '          end if\n'
    text_eval += '        else\n'
    text_eval += '          inputs_tau(1:2) = invars(2*ii-1:2*ii)\n'
    text_eval += '          if (numFeatures .gt. 0) then\n'
    text_eval += '            inputs_tau(3: 3+numFeatures-1) = features\n'
    text_eval += '          end if\n'
    text_eval += '        end if\n\n'

    text_eval += '        inputs_psi_a(1:2) = invars(2*ii-1:2*ii)\n'
    text_eval += '        if (numFeatures .gt. 0) then\n' 
    text_eval += '          inputs_psi_a(3: 3+numFeatures-1) = features\n'
    text_eval += '        end if\n\n'

    text_eval += '      do jj = 1, numMaxwell\n\n'

    text_eval += '        !=== Relaxation times \n' 
    text_eval += '        call derivatives_output(weights_tau(:,:,:,jj,ii), biases_tau(:,:,jj,ii), acts_t, inputs_tau,   &\n'
    text_eval += '                                nout_t, nin_t, L_t, want_hessian_tau, want_third_tau, &\n'
    text_eval += '                                J, H, T, y)\n'    
    text_eval += '        ! Collect outputs \n'
    text_eval += '        Tau(jj,ii) = y(1)*scale_tau(jj,ii) \n'
    text_eval += '        J_tau(:,:,jj,ii) = J*scale_tau(jj,ii) \n\n'

    text_eval += '        !=== Non-equilibrium free energy \n'
    text_eval += '        call derivatives_output(weights_psi_a(:,:,:,jj,ii), biases_psi_a(:,:,jj,ii), acts_p, inputs_psi_a,   &\n'
    text_eval += '                                nout_p, nin_p, L_neq, want_hessian_psi_a, want_third_psi_a, &\n'
    text_eval += '                                J, H, T, y)\n\n'

    text_eval += '        ! Correct for stress-free reference configuration\n'
    text_eval += '        J(1,1) = J(1,1) + alphas_a(jj,ii) \n'  
    text_eval += '        J(1,2) = J(1,2) + betas_a(jj,ii) \n\n'

    text_eval += '        ! Collect outputs \n'
    text_eval += '        J_Psi_a(:,:,jj,ii)     = J*L1(jj,ii) \n'
    text_eval += '        H_Psi_a(:,:,:,jj,ii)   = H*L1(jj,ii) \n'
    text_eval += '        T_Psi_a(:,:,:,:,jj,ii) = T*L1(jj,ii) \n\n'

    text_eval += '      end do\n'
    text_eval += '    end do\n\n'
    text_eval += '    deallocate(inputs_tau)\n'
    text_eval += '    deallocate(inputs_psi_a)\n\n'
    text_eval += '  end subroutine vCANN_non_equi \n\n'

    text_eval += '  ! ================================================================ \n'
    text_eval += '  ! Initialize the offsets for a stress-free reference configuration\n'
    text_eval += '  ! ================================================================ \n'
    text_eval += '  subroutine init_offset()\n\n'
    text_eval += '    use precision\n'
    text_eval += '    use deriv_recursive\n'
    text_eval += '    use parameters\n\n'

    text_eval += '    implicit none\n'
    text_eval += '    save\n'
            
    text_eval += '    ! Equilibrium offset\n'
    text_eval += '    real(dp), dimension(numTens) :: deltas\n'
    text_eval += '    real(dp), allocatable :: J_Psi_ref(:,:,:), H_Psi_ref(:,:,:,:)\n\n'

    text_eval += '    ! Non-equilibrium offset\n'
    text_eval += '    real(dp), dimension(numMaxwell, numTens) :: deltas_a\n'
    text_eval += '    real(dp), allocatable :: Tau(:,:), J_tau(:,:,:,:), J_Psi_a_ref(:,:,:,:), H_Psi_a_ref(:,:,:,:,:), T_Psi_a_ref(:,:,:,:,:,:)\n'

    text_eval += '    allocate(J_Psi_ref(1,2,numTens)); J_Psi_ref = 0.0_dp\n'
    text_eval += '    allocate(H_Psi_ref(1,2,2,numTens)); H_Psi_ref = 0.0_dp\n\n'

    text_eval += '    allocate(Tau(numMaxwell,numTens)); Tau = 0.0_dp\n'
    text_eval += '    allocate(J_Tau(1,2,numMaxwell,numTens)); J_Tau = 0.0_dp\n'
    text_eval += '    allocate(J_Psi_a_ref(1,2,numMaxwell,numTens)); J_Psi_a_ref = 0.0_dp\n'
    text_eval += '    allocate(H_Psi_a_ref(1,2,2,numMaxwell,numTens)); H_Psi_a_ref = 0.0_dp\n'
    text_eval += '    allocate(T_Psi_a_ref(1,2,2,2,numMaxwell,numTens)); T_Psi_a_ref = 0.0_dp\n\n'

    text_eval += '    ! Stress offset for equilibrium free energy\n'
    text_eval += '    call vCANN_equi(invarsRef, J_Psi_ref, H_Psi_ref)\n'
    text_eval += '    deltas = J_Psi_ref(1,1,:) - J_Psi_ref(1,2,:)\n'
    text_eval += '    alphas = max(-deltas, 0.0_dp)\n'
    text_eval += '    betas  = max(deltas, 0.0_dp)\n'
    text_eval += '    deallocate(J_Psi_ref); deallocate(H_Psi_ref)\n\n'
    text_eval += '    ! Stress offset for non-equilibrium free energy\n'
    text_eval += '    call vCANN_non_equi(invarsRef, invarsDotRef, Tau, J_Tau, J_Psi_a_ref, H_Psi_a_ref, T_Psi_a_ref)\n'
    text_eval += '    deltas_a = J_Psi_a_ref(1,1,:,:) - J_Psi_a_ref(1,2,:,:)\n'
    text_eval += '    alphas_a = max(-deltas_a, 0.0_dp)\n'
    text_eval += '    betas_a  = max(deltas_a, 0.0_dp)\n'
    text_eval += '    deallocate(Tau); deallocate(J_Tau); deallocate(J_Psi_a_ref); deallocate(H_Psi_a_ref); deallocate(T_Psi_a_ref)\n\n'
    
    text_eval += '  end subroutine init_offset\n\n'

    text_eval += 'end module vCANNs\n\n'

    text_4 += text_2_2 + text_init_psi_eq + text_init_psi_neq

    text_2 += text_init_collect + text_eval  
    
    #
    ### Subroutine Generalized Structural Tensors
    #
    text_9 = '\n\n'
    text_9 += '! ================================================================================================ !\n'
    text_9 += '! ===                        Subroutine Generalized Structural Tensors                         === !\n'
    text_9 += '! ================================================================================================ !\n\n'
    text_9 += 'subroutine vCANN_structural(extra_in, material_dirs, struc_weights) \n\n'  
    
    text_9 += '  use precision \n'    
    text_9 += '  use parameters \n'
    text_9 += '  !use activation_functions \n\n'   
    text_9 += '  implicit none \n\n'
    
    text_9 += '  ! formal variables \n'
    text_9 += '  ! ================================================================ \n'
    text_9 += '  real(dp), dimension(1,1), intent(in) :: extra_in                   ! input feature to structure learning \n'
    text_9 += '  real(dp), dimension(numDir,3), intent(out) :: material_dirs        ! preferred material directions \n'
    text_9 += '  real(dp), dimension(numTens,numDir+1), intent(out) ::  struc_weights ! scalar weights of the preferred material directions \n\n'
    
    if numDir == 0:
        text_9 += '\n'
        text_9 += '! Nothing to compute here since the material is isotropic!\n'
        
        text_3 = ''
        text_3 += '\n\n'
        text_3 += '  ! ================================================================ \n'
        text_3 += '  ! Generalized Structural Tensors (preferred directions and weights) \n' 
        text_3 += '  ! ================================================================ \n'
        text_3 += '  ! weights and biases are only introduced as dummies, since they have to be defined at some point. Otherwise modifying umat.f90 would be necessary! \n'
        text_3 += '  real(dp), dimension(numDir/2) :: theta = zero           ! angles specifying the orientation of a symmetric pair of preferred material directions in local xy-plane \n'
        text_3 += '  real(dp), dimension(numTens,numDir+1) :: d_weights = zero ! weights of the generalized structural tensors \n\n'

    elif numDir != 0 and numExtraStruc == 0:

        dir_model = model_fit.get_layer('model_dir')
        weight_model = model_fit.get_layer('model_w')
        
        material_dirs = dir_model(F_ref)[0,0]
        struc_weights = weight_model(F_ref)[0,0]
        
        dir_model_name =  dir_model.__class__.__name__
        
        if dir_model_name == 'dirModelSymFiber':
            if numDir % 2 != 0:
                raise ValueError('Only an even number of preferred fiber directions is allowed, since always two fibers at a time are assumed to form a symmetric pair.')
        
            text_9 += '  ! (Symmetric pairs of) preferred material directions, 1: axial, 2: circumferential, 3: radial \n'
            text_9 += '  ! ================================================================ \n'
            reshape_str = format_fortran_reshape('material_dirs', material_dirs, (numDir,3), indent="      ")
            text_9 += f"  real(dp), dimension({numDir},{3}), parameter :: {reshape_str} \n"            
            
            text_9 += '  ! Scalar weights of the generalized structural tensors \n'
            text_9 += '  ! ================================================================ \n'
            reshape_str = format_fortran_reshape('struc_weights', struc_weights, (numTens, numDir+1), indent="      ")
            text_9 += f"  real(dp), dimension({numTens},{numDir+1}), parameter :: {reshape_str} \n"

            # USE THE OUTPUT OF THE DIRECTION MODEL TO DEFINE FIXED DIRECTIONS INSTEAD OF USING ANGLES THETA
        
        if dir_model_name == 'dirModelOrtho':
            if numDir != 3:
                raise ValueError('For orthotropic materials, exactly three preferred material directions have to be defined.')
            
            text_9 += '  ! Preferred material directions (orthotropic material) \n'
            text_9 += '  ! ================================================================ \n'
            reshape_str = format_fortran_reshape('material_dirs', material_dirs, (numDir,3), indent="      ")
            text_9 += f"  real(dp), dimension({numDir},{3}), parameter :: {reshape_str} \n"

            struc_weights = tf.concat([tf.zeros((numTens,1), dtype=struc_weights.dtype), struc_weights], axis=1) # add zero weight for isotropic part 1/3*L_0
            text_9 += '  ! Scalar weights of the generalized structural tensors \n'
            text_9 += '  ! ================================================================ \n'
            reshape_str= format_fortran_reshape('struc_weights', tf.transpose(struc_weights), (numTens, numDir+1), indent="      ")
            text_9 += f"  real(dp), dimension({numTens},{numDir+1}), parameter :: {reshape_str} \n"

        text_3 = ''

    elif numDir != 0 and numExtraStruc != 0:
        
        # Variables for preferred material directions and scalar weighting factors
        text_9 += '\n'
        text_9 += '! internal variables\n'
        text_9 += '! ================================================================'
        
        text_3 = ''
        text_3 += '\n\n'
        text_3 += '! Generalized Structural Tensors (preferred directions and weights) \n' 
        text_3 += '! ================================================================ \n'
                   
        num = [numDir, numTens]
        prefixes = ['dir', 'w']
        title = ['Direction', 'Scalar weighting factors']
        for p, n, t in zip(prefixes, num, title):
            model = model_fit.get_layer('model_{}'.format(p))
            for kk in range(1,n+1):
                text_9   += '\n! {} {} \n'.format(t, kk)
                text_3 += '\n! {} {} \n'.format(t, kk)
                layers = [l for l in model.layers if '{}_{}'.format(p, kk) in l.name]
                numLayers = len(layers)
                for ii, l in enumerate(layers):
                    text_9   += '! Layer {} \n'.format((ii)%numLayers+1)
                    text_3 += '! Layer {} \n'.format((ii)%numLayers+1)
                    
                    text_3 += 'real(dp), dimension({},{}) :: W_{} \n'.format(l.kernel.shape[0], l.kernel.shape[1], l.name)
                    if l.use_bias:
                        text_3 += 'real(dp), dimension({}) :: b_{} \n'.format(l.bias.shape[0], l.name)
    
                    text_9 += 'real(dp), dimension(1,{}) :: z_{} \n'.format(l.kernel.shape[1], l.name)
                    text_9 += 'real(dp), dimension(1,{}) :: x_{} \n'.format(l.kernel.shape[1], l.name)
                               
        text_9 += '\n\n'
        text_9 += '! calculate preferred directions and weighting factors \n'
        text_9 += '! ================================================================ \n\n'
    
        num = [numDir, numTens]
        prefixes = ['dir', 'w']
        title = ['Direction l', 'Scalar weighting factors w']    
        for p, n, t in zip(prefixes, num, title):
            model = model_fit.get_layer('model_{}'.format(p))
            for tt in range(1,n+1):
                text_9 += '! {}_{} \n'.format(t, tt)
                text_9 += '! ================================================================ \n'
                layers = [l for l in model.layers if '{}_{}'.format(p, tt) in l.name]
                numLayers = len(layers)
                for ii, l in enumerate(layers,1):
                    lCount = (ii-1) % numLayers+1
                    if (ii-1) % numLayers == 0:
                        text_9 += 'z_{}_{}_{} = matmul(extra_in, W_{}_{}_{}) + b_{}_{}_{} \n'.format(p, tt, lCount, p, tt, lCount, p, tt, lCount)
                    else:
                        if l.use_bias:
                            text_9 += 'z_{}_{}_{} = matmul(x_{}_{}_{}, W_{}_{}_{}) + b_{}_{}_{} \n'.format(p, tt, lCount, p, tt, lCount-1, p, tt, lCount, p, tt, lCount)
                        else:
                            text_9 += 'z_{}_{}_{} = matmul(x_{}_{}_{}, W_{}_{}_{}) \n'.format(p, tt, lCount, p, tt, lCount-1, p, tt, lCount)
                        
                    text_9 += 'x_{}_{}_{} = activation(z_{}_{}_{}, "{}") \n'.format(p, tt, lCount, p, tt, lCount, l.activation.__name__)
                    
                    if (ii-1) % numLayers == (numLayers-1):
                        if p == 'dir':
                            text_9 += '\n! Make preferred material directions unit vectors \n'
                            text_9 += '! ================================================================ \n'
                            text_9 += 'x_{}_{}_{}  = x_{}_{}_{} / norm2(x_{}_{}_{})  \n'.format(p, tt, lCount, p, tt, lCount, p, tt, lCount)
                            text_9 += 'material_dirs({}:{},:) = x_{}_{}_{} \n'.format(tt, tt, p, tt, lCount)
                        if p == 'w':
                            text_9 += '\n! Normalize direction weights such that they are in the interval [0,1] \n'
                            text_9 += '! ================================================================ \n'
                            text_9 += 'x_{}_{}_{} = x_{}_{}_{} / sum(x_{}_{}_{})\n'.format(p, tt, lCount, p, tt, lCount, p, tt, lCount)                    
                            text_9 += 'dir_weights({}:{},:) = x_{}_{}_{} \n'.format(tt, tt, p, tt, lCount)                    
                text_9 += '\n'
        else:
            raise ValueError('Missmatch between number of preferred material directions and the extra input for the generalized structural tensors.')
            
        
    text_9 += '\nend subroutine vCANN_structural\n\n'
    
    
    
    with open("..\\Abaqus\\\\umat.f90", 'r', encoding='utf-8-sig') as file:
        umat = file.read()
    
    # Load and initialize network weights
    text_10 = '\n\n'   
    text_10 += '! ================================================================================================ !\n'
    text_10 += '! ===                                   Subroutine UEXTERNALDB                                 === !\n'
    text_10 += '! ================================================================================================ !\n\n'
    
    text_10 += 'subroutine UEXTERNALDB(LOP,LRESTART,TIME,DTIME,KSTEP,KINC) \n\n'
    
    text_10 += '! Loads the vCANN kernels and biases and stores them accessible for each subroutine in the "parameters" module \n\n'
    
    text_10 += '  use parameters \n'
    text_10 += '  use vCANNs \n'
    text_10 += '  implicit none \n\n'
                
    text_10 += '  ! formal variables \n'
    text_10 += '  ! ================================================================ \n'
    text_10 += '  integer :: LOP, LRESTART, KSTEP, KINC \n'
    text_10 += '  double precision :: TIME, DTIME \n'
    text_10 += '  dimension :: TIME(2) \n'
      
    text_10 += '\n'    
    text_10 += '!  do while(myVar /= 999) \n'
    text_10 += '!      myVar = 1 \n'
    text_10 += '!  end do \n\n'
      
    text_10 += '  if ((LOP == 0) .or. (LOP == 4)) then \n'
    text_10 += '    call init_weights_eq() \n'
    text_10 += '    call init_weights_neq() \n'
    text_10 += '    call init_offset() \n'
    text_10 += '  end if \n\n'

    text_10 += 'end subroutine UEXTERNALDB'
    
    
    text_10 += '\n\n'
    text_10 += '! ================================================================================================ !\n'
    text_10 += '! ===                                     Subroutine SDVINI                                    === !\n'
    text_10 += '! ================================================================================================ !\n\n'
    
    text_10 += 'subroutine SDVINI(STATEV,COORDS,NSTATV,NCRDS,NOEL,NPT,LAYER,KSPT) \n\n'
    
    text_10 += '! Calculates the generalized structural tensors at the beginning of an analysis for each element'
    
    text_10 += 'use parameters \n'

    text_10 += 'implicit none \n'
    
    text_10 += '! formal variables \n'
    text_10 += '! ================================================================ \n'
    text_10 += 'integer :: NSTATV, NCRDS, NOEL, NPT, LAYER, KSPT \n'
    text_10 += 'double precision :: STATEV, COORDS \n'
    text_10 += 'dimension STATEV(NSTATV), COORDS(NCRDS) \n'

    text_10 += 'END subroutine SDVINI \n\n'
    
    
    ### UMAT
    finalText = text_0 + text_3 + text_4 + text_1 + text_2 + umat + text_9 + text_10
    
    if save_local:
        file_format = ".for"
    else:
        file_format = ".f90"
        
    file = umat_folder + "\\umat_vCANN" + file_format 
    with open(file, 'w') as f:
        f.write(finalText)        
           
