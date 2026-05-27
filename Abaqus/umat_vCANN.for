!dec$ freeform 

include 'precision.f90' 
include 'umat_statev_utils.f90' 
include 'FFNN_derivatives.f90' 

module parameters 

  use precision 
  implicit none 
  save 

  integer, parameter :: numTens = 1 ! number of generalized structural tensors 
  integer, parameter :: numDir = 0 ! number of preferred material directions 
  integer, parameter :: numMaxwell = 3 ! number of Maxwell elements per generalized structural tensor 
  integer, parameter :: numFeatures = 0 ! number of elements of the feature vector 
  logical, parameter :: rateDependent = .FALSE. ! flag for rate dependency of relaxation times 

  real(dp) :: zero, one, two, three, four, six 
  parameter(zero=0.0_dp, one=1.0_dp, two=2.0_dp, three=3.0_dp, four=4.0_dp, six=6.0_dp) 

  ! Identity matrix 
  real(dp), dimension(3,3), parameter :: eye &  
      = reshape((/one,zero,zero,zero,one,zero,zero,zero,one/),(/3,3/))  

  ! Generalized invariants in the reference configuration 
  real(dp), dimension(2*numTens), parameter :: invarsRef = one  
  real(dp), dimension(2*numTens+1), parameter :: invarsDotRef = zero  

  real(dp), parameter, dimension(numFeatures) :: features = 1.0_dp  ! feature vector 
  real(dp), parameter, dimension(1,1) :: extra_struc = (/ 1.0_dp /) ! dummy input 

  ! UMAT flags 
  logical, parameter :: use_numerical_tangent = .FALSE. ! flag for choosing numerical tangent 
  real(dp), parameter :: epsilon = 2.e-8_dp ! perturbation size for numerical tangent

  logical, parameter :: use_shell = .FALSE. ! flag if shell elements are used 
  logical, parameter :: local_csys = .FALSE. ! flag if a local coordinate system is used 


  ! ================================================================ 
  ! Generalized Structural Tensors (preferred directions and weights) 
  ! ================================================================ 
  ! weights and biases are only introduced as dummies, since they have to be defined at some point. Otherwise modifying umat.f90 would be necessary! 
  real(dp), dimension(numDir/2) :: theta = zero           ! angles specifying the orientation of a symmetric pair of preferred material directions in local xy-plane 
  real(dp), dimension(numTens,numDir+1) :: d_weights = zero ! weights of the generalized structural tensors 

  ! ================================================================ 
  ! Equilibrium free energy 
  ! ================================================================ 
  real(dp), parameter :: kappa= zero ! volumetric contribution - penalty parameter (only for nearly-incompressible material) 

  integer, parameter :: L_eq = 2
  integer, parameter :: max_in=10, max_out=10
  integer, dimension(L_eq), parameter :: nin = [ 2, 10 ], nout = [ 10, 1 ], acts = [ 3, 4 ]
  real(dp), dimension(max_out, max_in, L_eq, numTens) :: weights = 0.0_dp
  real(dp), dimension(max_out, L_eq, numTens) :: biases = 0.0_dp

  ! Stress offset equilibrium free energy 
  real(dp), dimension(numTens) :: alphas = 0.0_dp ! initialized by init_offset()
  real(dp), dimension(numTens) :: betas = 0.0_dp 


  ! Equilibrium free energy contribution 1 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Psi_1_1 = reshape([ &
      7.3054053227592124e+02_dp, 4.4647671488049008e+02_dp,  &
      3.8790900476044305e+02_dp, 2.0949863692779223e+02_dp,  &
      5.8382164356443539e+01_dp, 6.9150314082668240e+02_dp,  &
      5.4220337402672874e+02_dp, 3.5182991312071414e+02_dp,  &
      3.5093681654671826e+02_dp, 2.2168229434051247e+02_dp,  &
      2.3359844302480468e+02_dp, 4.9190869136350113e+02_dp,  &
      4.5749414957965905e+02_dp, 2.8610217312828127e+02_dp,  &
      8.5748349310205704e-12_dp, 2.6282412891476656e+02_dp,  &
      0.0000000000000000e+00_dp, 1.9339899600030634e+02_dp,  &
      3.0751966689764345e+02_dp, 1.1163336621397487e+02_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Psi_1_1 = [ 4.9743328261776952e+02_dp, 4.0734979102036579e+02_dp, 3.5742610511461180e+02_dp, 1.9065746158808847e+02_dp, 2.4332059615094110e+01_dp, 4.5144881508001828e+02_dp, -5.4289650018802490e+02_dp, 2.4042165973218687e+02_dp, 2.7691860623842541e+02_dp, 1.2596521996485750e+02_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Psi_1_2 = reshape([ &
      0.0000000000000000e+00_dp, 0.0000000000000000e+00_dp,  &
      0.0000000000000000e+00_dp, 0.0000000000000000e+00_dp,  &
      0.0000000000000000e+00_dp, 0.0000000000000000e+00_dp,  &
      1.9918689534800482e-02_dp, 0.0000000000000000e+00_dp,  &
      0.0000000000000000e+00_dp, 0.0000000000000000e+00_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Psi_1_2 = 0.0_dp 

  ! ================================================================ 
  ! Non-equilibrium free energy 
  ! ================================================================ 
  integer, parameter :: L_t = 2
  integer, parameter :: L_neq = 2

  integer, parameter :: max_in_tau = 10, max_out_tau = 10

  integer, parameter :: max_in_psi_a = 10, max_out_psi_a = 10
  integer, dimension(L_t), parameter :: nin_t = [ 2, 10 ], nout_t = [ 10, 1 ], acts_t = [ 3, 3 ]
  integer, dimension(L_neq), parameter :: nin_p = [ 2, 10 ], nout_p = [ 10, 1 ], acts_p = [ 6, 4 ]

  ! Network weights and biases
  real(dp), dimension(max_out_tau, max_in_tau, L_t, numMaxwell, numTens) :: weights_Tau = 0.0_dp
  real(dp), dimension(max_out_tau, L_t, numMaxwell, numTens) :: biases_Tau = 0.0_dp

  real(dp), dimension(max_out_psi_a, max_in_psi_a, L_neq, numMaxwell, numTens) :: weights_Psi_a = 0.0_dp
  real(dp), dimension(max_out_psi_a, L_neq, numMaxwell, numTens) :: biases_Psi_a = 0.0_dp

  ! Stress offset non-equilibrium free energy 
  real(dp), dimension(numMaxwell,numTens) :: alphas_a = 0.0_dp ! initialized by init_offset() 
  real(dp), dimension(numMaxwell,numTens) :: betas_a = 0.0_dp 

  ! Time scaling 
  real(dp), dimension(numMaxwell,numTens), parameter :: scale_tau = reshape([ &
      1.0000000000000000e+00_dp, 1.0000000000000000e+01_dp,  &
      1.0000000000000000e+02_dp &
  ], [3, 1]) 

  ! L1 sparsity 
  real(dp), dimension(numMaxwell,numTens), parameter :: L1 = reshape([ &
      1.0000000000000000e+00_dp, 1.0000000000000000e+00_dp,  &
      1.0000000000000000e+00_dp &
  ], [3, 1]) 


  ! ================================================================ 
  ! Generalized Maxwell model 1 
  ! Non-equilibrium free energy Psi_a_1 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Tau_1_1_1 = reshape([ &
      -1.0615497059951564e+00_dp, 5.5578240821333402e-01_dp,  &
      -2.0246227069437559e+00_dp, 1.8180448304972263e+00_dp,  &
      5.4816450750817980e-01_dp, -2.0514006458426510e+00_dp,  &
      4.4946506356685963e-02_dp, -1.5902969675108403e+00_dp,  &
      3.1011110715457429e-01_dp, 2.3859082990810815e-01_dp,  &
      5.4265550872521107e-01_dp, 4.9481295389693675e-01_dp,  &
      1.6948889753960678e-01_dp, 1.5473459600302431e+00_dp,  &
      -1.4223644335441818e-01_dp, -1.7291133820188058e+00_dp,  &
      -9.2510851365026003e-01_dp, -1.0088462371627891e+00_dp,  &
      4.8204010634703309e-01_dp, 1.4584032882265050e+00_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Tau_1_1_1 = [ -1.8394445952974789e-01_dp, 4.7182278187740168e-01_dp, -1.2100600249456028e-01_dp, -2.6478939255802619e-01_dp, 9.9809712373056761e-02_dp, -3.8603887175013663e-01_dp, 1.0392332173611817e-01_dp, -4.7103116968084541e-01_dp, 3.1152727503985939e-01_dp, -2.1430625817168628e-01_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Tau_1_1_2 = reshape([ &
      2.9010462105704077e-01_dp, -6.3708653243436042e-01_dp,  &
      -2.3389119760559993e-01_dp, -1.8806017463778837e+00_dp,  &
      -4.2415369582727169e-01_dp, 2.7042319260121112e-01_dp,  &
      1.4817259631909158e-01_dp, -4.0890653457135634e-02_dp,  &
      -2.8158143873001984e-01_dp, -1.0932238048414697e+00_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Tau_1_1_2 = [ -1.4711981506133871e+00_dp ]

  ! Non-equilibrium free energy Psi_a_2 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Tau_1_2_1 = reshape([ &
      1.7845066999701445e-01_dp, 3.3654229658468149e-01_dp,  &
      1.2849104350772256e+00_dp, -1.0150072741093270e+00_dp,  &
      -5.2673182063641366e-01_dp, 2.7287719398846971e-01_dp,  &
      -1.4758834382986569e-01_dp, -4.7497754064049241e-01_dp,  &
      2.9567029084094143e-01_dp, 1.4248755260260604e+00_dp,  &
      8.1766732519295859e-01_dp, 7.1396226392512041e-01_dp,  &
      3.5339594992567763e-01_dp, -1.6629050904206544e+00_dp,  &
      1.5560910626249962e+00_dp, 1.1320348643282052e-01_dp,  &
      1.2607463966920043e+00_dp, 1.5257178558843791e+00_dp,  &
      -2.5981479489981454e+00_dp, 2.6005787731526941e-01_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Tau_1_2_1 = [ -1.4629633944857087e-01_dp, -7.3592203465856446e-02_dp, 1.0601265678391300e-01_dp, -3.6475521997743071e-01_dp, -2.1993280350921390e-01_dp, -5.8274801274988663e-01_dp, -1.5488143240614896e-01_dp, 2.2266256061073250e-01_dp, -3.3217378144376469e-01_dp, 1.1886624098737386e+00_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Tau_1_2_2 = reshape([ &
      -1.3874722212547694e-01_dp, -1.1637199609008612e-01_dp,  &
      -5.0836413410706771e-01_dp, -5.7081047912184624e-01_dp,  &
      1.3486150585250650e-01_dp, -1.5627184724733939e-01_dp,  &
      1.9487632091370899e-01_dp, -1.7512434176687974e-01_dp,  &
      2.3650117893706113e-01_dp, 1.9105692234508864e-01_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Tau_1_2_2 = [ 1.2990215114534234e+00_dp ]

  ! Non-equilibrium free energy Psi_a_3 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Tau_1_3_1 = reshape([ &
      -1.3087111823989606e+00_dp, 7.3044018739924610e-01_dp,  &
      5.2857950949288734e-01_dp, 9.4805011670286055e-01_dp,  &
      -1.0599419207360132e+00_dp, 8.1573059159951655e-01_dp,  &
      9.2056779450352322e-01_dp, -1.9113245511798951e+00_dp,  &
      6.2559858819138885e+00_dp, -5.6928738716452076e-01_dp,  &
      1.2025579801362336e+00_dp, -1.0938289934131362e+00_dp,  &
      2.3868137622103891e+00_dp, -1.7761421181539585e+00_dp,  &
      -5.1872241975500055e-01_dp, -2.6538318251435450e-01_dp,  &
      1.2958892870330709e+00_dp, 9.7631583344002026e-01_dp,  &
      5.0530157928664003e+00_dp, 1.5233314828352446e-01_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Tau_1_3_1 = [ 3.2319517796270008e+00_dp, 2.4629537183427357e+00_dp, -8.9054133439250704e-01_dp, 1.0018143074105623e+00_dp, -2.8908550558525317e-03_dp, 5.8937232939785438e-03_dp, -6.1062285531654381e-01_dp, 3.5504240374558854e-01_dp, -7.3866719198265098e+00_dp, 2.0669438473779009e+00_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Tau_1_3_2 = reshape([ &
      4.3765603373795390e+00_dp, 3.5065492245321046e+00_dp,  &
      -5.1006106646426519e-01_dp, 1.9758842191232655e+00_dp,  &
      1.1322410423197313e+00_dp, 1.2348503676442484e+00_dp,  &
      -5.8106421072786962e-01_dp, 1.6458580436135744e+00_dp,  &
      -9.7098065604025656e+00_dp, 3.3593904424620553e+00_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Tau_1_3_2 = [ 4.9393020301842681e+00_dp ]

  ! Non-equilibrium free energy Psi_a_1 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Psi_a_1_1_1 = reshape([ &
      1.5845758181968639e+00_dp, 1.7584530686315042e-01_dp,  &
      -1.1474135320041641e+00_dp, 6.7383954549892189e-01_dp,  &
      6.4398419737290047e-01_dp, 1.1254357234963304e-01_dp,  &
      6.2599785244570882e-02_dp, 4.5152198554174938e-01_dp,  &
      4.2745000757305154e-02_dp, 1.4378894721415989e+00_dp,  &
      9.8637970486039250e-01_dp, 1.4903277702537674e+00_dp,  &
      1.6550600694118620e+00_dp, -2.5504009088931268e-01_dp,  &
      -4.8158850205664167e-01_dp, -1.0400625383335891e+00_dp,  &
      -8.7142236532676265e-01_dp, 2.2785256645419941e-01_dp,  &
      -5.4802700730327591e-01_dp, 2.3992220769653833e-01_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Psi_a_1_1_1 = [ -4.6721409028351008e-01_dp, 5.0571410079450341e-01_dp, 4.8552311751980504e-01_dp, 6.1393325207082211e-02_dp, 1.1830002281247810e-01_dp, 3.5297039398179642e-01_dp, -4.8910751009280001e-01_dp, 3.7660860598052853e-01_dp, -1.8482856670388148e-01_dp, 1.9550189771242793e-01_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Psi_a_1_1_2 = reshape([ &
      -9.9760303843206788e-01_dp, -7.1062831947295901e-01_dp,  &
      -1.3038611347866222e-01_dp, 6.3539050894335580e-01_dp,  &
      7.5403469214172958e-01_dp, -3.2167908418369612e-01_dp,  &
      -1.4072741053252127e-01_dp, -5.9116210412582737e-01_dp,  &
      -5.5426105593682162e-01_dp, -7.3114476388826433e-01_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Psi_a_1_1_2 = 0.0_dp 

  ! Non-equilibrium free energy Psi_a_2 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Psi_a_1_2_1 = reshape([ &
      -1.5691834527680264e+00_dp, -3.0678688586204811e-01_dp,  &
      -3.0650178594900529e-01_dp, -2.9010301229480301e-01_dp,  &
      5.7317156476389526e+00_dp, 4.7303071939524735e+00_dp,  &
      -1.2587369892758205e+00_dp, -2.1054299710262843e+00_dp,  &
      -5.2125450408161829e+00_dp, -1.2959437632921365e+00_dp,  &
      -5.7923719545307391e+00_dp, 2.5314972501482935e+00_dp,  &
      1.2693913751421613e+00_dp, -6.5231071626602266e-01_dp,  &
      -1.6766696116957014e-01_dp, -9.7463706630630947e-02_dp,  &
      5.0218458895578454e-03_dp, 1.7118388600886145e-01_dp,  &
      1.1654696113688394e-01_dp, 1.0651441151361256e+00_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Psi_a_1_2_1 = [ 5.5413926353170169e+00_dp, 5.0656830768758465e-01_dp, -2.6518291174823055e-01_dp, -2.9015944256356530e-01_dp, -3.1035390400368832e+00_dp, 1.0348169346657137e+00_dp, 4.2559525576556906e-01_dp, 1.7748717590745928e+00_dp, 3.6940044728946431e+00_dp, 5.1057532001754624e-01_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Psi_a_1_2_2 = reshape([ &
      -3.8921884673441833e+00_dp, -1.4757088589508549e-01_dp,  &
      1.0075105736951195e+00_dp, -2.6874071886218842e-01_dp,  &
      2.3890791532852882e+00_dp, -3.5607880229220821e+00_dp,  &
      -9.8372172148485182e-01_dp, -1.9311081779571275e+00_dp,  &
      -3.5010918049131758e+00_dp, -1.0458532452550351e+00_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Psi_a_1_2_2 = 0.0_dp 

  ! Non-equilibrium free energy Psi_a_3 
  ! ================================================================ 
  ! Layer 1 
  real(dp), dimension(10,2), parameter :: W_Psi_a_1_3_1 = reshape([ &
      -2.0118547507801150e+00_dp, -6.0428662039400727e-01_dp,  &
      -1.2016174554478450e-01_dp, -1.7097305271525416e+00_dp,  &
      5.9694109610973634e-01_dp, -2.0205523059742361e+00_dp,  &
      8.6884563936215464e-01_dp, -3.6714611637539631e+00_dp,  &
      4.1188016794631899e-01_dp, 5.3947067899062440e-01_dp,  &
      -4.8718769413469393e-01_dp, -2.1677619752568527e+00_dp,  &
      -1.9282418675086235e+00_dp, -5.8466730515259779e-01_dp,  &
      -1.5055125880240112e+00_dp, 2.5171623306269200e+00_dp,  &
      3.5651034720601955e+00_dp, 3.1832367239399351e+00_dp,  &
      -2.1867735594936786e+00_dp, 1.2056464537463236e+00_dp &
  ], [2, 10]) 
  real(dp), dimension(10), parameter :: b_Psi_a_1_3_1 = [ 3.3930867324146874e-01_dp, 2.1327807548305109e+00_dp, 1.1648278617465349e+00_dp, 2.5760254671246363e-01_dp, -2.7177228058516822e-01_dp, 8.8402697686282172e-01_dp, -5.4120607449559699e-01_dp, 1.5743351614954664e+00_dp, 3.3275802043631603e-01_dp, 3.7819194060550082e-01_dp ]

  ! Layer 2 
  real(dp), dimension(1,10), parameter :: W_Psi_a_1_3_2 = reshape([ &
      -2.6541056976612856e+00_dp, -2.1116346923361262e+00_dp,  &
      -1.7242801616062391e+00_dp, 6.3919783096456587e-02_dp,  &
      -9.2479643203754447e-01_dp, 2.3355106002285075e+00_dp,  &
      -1.3623415902040537e+00_dp, 3.4403831284646302e+00_dp,  &
      -1.5245277396646675e+00_dp, -7.3337231604955894e-02_dp &
  ], [10, 1]) 
  real(dp), dimension(1), parameter :: b_Psi_a_1_3_2 = 0.0_dp 

contains 

  subroutine init_weights_eq()
  ! Initialize weights before analysis starts 
    implicit none 
    save 

    weights(1:10, 1:2, 1, 1) = W_Psi_1_1 
    biases(1:10, 1, 1) = b_Psi_1_1 

    weights(1:1, 1:10, 2, 1) = W_Psi_1_2 
    biases(1:1, 2, 1) = b_Psi_1_2 

  end subroutine init_weights_eq 

  subroutine init_weights_neq()
  ! Initialize weights before analysis starts 
    implicit none 
    save 

    ! ================================================================ 
    ! Generalized Maxwell model 1 

    ! Non-equilibrium free energy Psi_a_1 
    ! ================================================================ 
    weights_Tau(1:10, 1:2, 1, 1, 1) = W_Tau_1_1_1 
    biases_Tau(1:10, 1, 1, 1) = b_Tau_1_1_1 
    weights_Tau(1:1, 1:10, 2, 1, 1) = W_Tau_1_1_2 
    biases_Tau(1:1, 2, 1, 1) = b_Tau_1_1_2 

    ! Non-equilibrium free energy Psi_a_2 
    ! ================================================================ 
    weights_Tau(1:10, 1:2, 1, 2, 1) = W_Tau_1_2_1 
    biases_Tau(1:10, 1, 2, 1) = b_Tau_1_2_1 
    weights_Tau(1:1, 1:10, 2, 2, 1) = W_Tau_1_2_2 
    biases_Tau(1:1, 2, 2, 1) = b_Tau_1_2_2 

    ! Non-equilibrium free energy Psi_a_3 
    ! ================================================================ 
    weights_Tau(1:10, 1:2, 1, 3, 1) = W_Tau_1_3_1 
    biases_Tau(1:10, 1, 3, 1) = b_Tau_1_3_1 
    weights_Tau(1:1, 1:10, 2, 3, 1) = W_Tau_1_3_2 
    biases_Tau(1:1, 2, 3, 1) = b_Tau_1_3_2 

    ! ================================================================ 
    ! Generalized Maxwell model 1 

    ! Non-equilibrium free energy Psi_a_1 
    ! ================================================================ 
    weights_Psi_a(1:10, 1:2, 1, 1, 1) = W_Psi_a_1_1_1 
    biases_Psi_a(1:10, 1, 1, 1) = b_Psi_a_1_1_1 
    weights_Psi_a(1:1, 1:10, 2, 1, 1) = W_Psi_a_1_1_2 
    biases_Psi_a(1:1, 2, 1, 1) = b_Psi_a_1_1_2 

    ! Non-equilibrium free energy Psi_a_2 
    ! ================================================================ 
    weights_Psi_a(1:10, 1:2, 1, 2, 1) = W_Psi_a_1_2_1 
    biases_Psi_a(1:10, 1, 2, 1) = b_Psi_a_1_2_1 
    weights_Psi_a(1:1, 1:10, 2, 2, 1) = W_Psi_a_1_2_2 
    biases_Psi_a(1:1, 2, 2, 1) = b_Psi_a_1_2_2 

    ! Non-equilibrium free energy Psi_a_3 
    ! ================================================================ 
    weights_Psi_a(1:10, 1:2, 1, 3, 1) = W_Psi_a_1_3_1 
    biases_Psi_a(1:10, 1, 3, 1) = b_Psi_a_1_3_1 
    weights_Psi_a(1:1, 1:10, 2, 3, 1) = W_Psi_a_1_3_2 
    biases_Psi_a(1:1, 2, 3, 1) = b_Psi_a_1_3_2 

  end subroutine init_weights_neq 

end module parameters 

! ================================================================ 

module vCANNs 

  use precision 
  use parameters 
  implicit none 

  private
  public :: vCANN_equi, vCANN_non_equi, init_offset

  contains 

  ! ================================================================ 
  ! Equilibrium free energy 
  ! ================================================================ 
  subroutine vCANN_equi(invars, J_Psi, H_Psi)

    use precision
    use deriv_recursive
    use parameters
    implicit none 

    ! Network inputs
    real(dp), dimension(2*numTens), intent(in) :: invars
    real(dp), dimension(2+numFeatures) :: inputs

    ! Network outputs
    real(dp), allocatable, intent(out) :: J_Psi(:,:,:), H_Psi(:,:,:,:)
    real(dp), allocatable :: Psi(:), J(:,:), H(:,:,:), T(:,:,:,:)
    logical, parameter :: want_hessian=.True., want_third=.False.
    integer :: ii

    allocate(J_Psi(nout(L_eq),nin(1),numTens)); J_Psi = 0.0_dp
    allocate(H_Psi(nout(L_eq),nin(1),nin(1),numTens)); H_Psi = 0.0_dp

    ! ================================================================ 
    ! Evaluate ANN and compute derivatives 
    ! ================================================================ 
    do ii = 1, numTens

        if (numFeatures .GT. 0) then 
            inputs(1:2) = invars(2*ii-1:2*ii)
            inputs(3:3+numFeatures-1) = features
        else 
            inputs =  invars(2*ii-1:2*ii)
        end if 

        call derivatives_output(weights(:,:,:,ii), biases(:,:,ii), acts, inputs,   &
                                nout, nin, L_eq, want_hessian, want_third, &
                                J, H, T, Psi)

        ! Correct for stress-free reference configuration 
        J(1,1) = J(1,1) + alphas(ii) 
        J(1,2) = J(1,2) + betas(ii) 

        ! Collect outputs 
        J_Psi(:,:,ii) = J 
        H_Psi(:,:,:,ii) = H 

    end do

  end subroutine vCANN_equi 

  ! ================================================================ 
  ! Non-equilibrium free energy and relaxation time
  ! ================================================================ 
  subroutine vCANN_non_equi(invars, invars_dot, Tau, J_Tau, J_Psi_a, H_Psi_a, T_Psi_a)

    use precision
    use deriv_recursive
    use parameters
    implicit none 

    ! Network inputs
    real(dp), intent(in) :: invars(:) ! generalized invariants
    real(dp), intent(in) :: invars_dot(:) ! generalized invariants of the material time derivative of the RCG tensor
    real(dp), allocatable :: inputs_tau(:)
    real(dp), allocatable :: inputs_psi_a(:)

    ! Network outputs
    real(dp), allocatable, intent(out) :: Tau(:,:) 
    real(dp), allocatable, intent(out) :: J_tau(:,:,:,:) 
    real(dp), allocatable, intent(out) :: J_Psi_a(:,:,:,:), H_Psi_a(:,:,:,:,:), T_Psi_a(:,:,:,:,:,:)
    real(dp), allocatable :: J(:,:), H(:,:,:), T(:,:,:,:)
    real(dp), allocatable :: y(:)

    logical, parameter :: want_hessian_tau=.False., want_third_tau=.False.
    logical, parameter :: want_hessian_psi_a=.True., want_third_psi_a=.True.
    integer :: ii, jj

    allocate(Tau(numMaxwell,numTens)); Tau = 0.0_dp
    allocate(J_Tau(nout_t(L_t),nin_t(1),numMaxwell,numTens)); J_Tau = 0.0_dp

    allocate(J_Psi_a(nout_p(L_neq),nin_p(1),numMaxwell,numTens)); J_Psi_a = 0.0_dp
    allocate(H_Psi_a(nout_p(L_neq),nin_p(1),nin_p(1),numMaxwell,numTens)); H_Psi_a = 0.0_dp
    allocate(T_Psi_a(nout_p(L_neq),nin_p(1),nin_p(1),nin_p(1),numMaxwell,numTens)); T_Psi_a = 0.0_dp

    ! ================================================================ 
    ! Evaluate ANN and compute derivatives 
    ! ================================================================ 
    if (rateDependent == .TRUE.) then 
      allocate(inputs_tau(5 + numFeatures)); inputs_tau = 0.0_dp
    else
      allocate(inputs_tau(2 + numFeatures)); inputs_tau = 0.0_dp
    end if

    allocate(inputs_psi_a(2 + numFeatures)); inputs_psi_a = 0.0_dp

    do ii = 1, numTens

        if (rateDependent == .TRUE.) then
          inputs_tau(1:2) = invars(2*ii-1:2*ii)
          inputs_tau(3:4) = invars_dot(2*ii-1:2*ii)
          inputs_tau(5) = invars_dot(2*numTens+1)
          if (numFeatures .gt. 0) then
            inputs_tau(6: 6+numFeatures-1) = features
          end if
        else
          inputs_tau(1:2) = invars(2*ii-1:2*ii)
          if (numFeatures .gt. 0) then
            inputs_tau(3: 3+numFeatures-1) = features
          end if
        end if

        inputs_psi_a(1:2) = invars(2*ii-1:2*ii)
        if (numFeatures .gt. 0) then
          inputs_psi_a(3: 3+numFeatures-1) = features
        end if

      do jj = 1, numMaxwell

        !=== Relaxation times 
        call derivatives_output(weights_tau(:,:,:,jj,ii), biases_tau(:,:,jj,ii), acts_t, inputs_tau,   &
                                nout_t, nin_t, L_t, want_hessian_tau, want_third_tau, &
                                J, H, T, y)
        ! Collect outputs 
        Tau(jj,ii) = y(1)*scale_tau(jj,ii) 
        J_tau(:,:,jj,ii) = J*scale_tau(jj,ii) 

        !=== Non-equilibrium free energy 
        call derivatives_output(weights_psi_a(:,:,:,jj,ii), biases_psi_a(:,:,jj,ii), acts_p, inputs_psi_a,   &
                                nout_p, nin_p, L_neq, want_hessian_psi_a, want_third_psi_a, &
                                J, H, T, y)

        ! Correct for stress-free reference configuration
        J(1,1) = J(1,1) + alphas_a(jj,ii) 
        J(1,2) = J(1,2) + betas_a(jj,ii) 

        ! Collect outputs 
        J_Psi_a(:,:,jj,ii)     = J*L1(jj,ii) 
        H_Psi_a(:,:,:,jj,ii)   = H*L1(jj,ii) 
        T_Psi_a(:,:,:,:,jj,ii) = T*L1(jj,ii) 

      end do
    end do

    deallocate(inputs_tau)
    deallocate(inputs_psi_a)

  end subroutine vCANN_non_equi 

  ! ================================================================ 
  ! Initialize the offsets for a stress-free reference configuration
  ! ================================================================ 
  subroutine init_offset()

    use precision
    use deriv_recursive
    use parameters

    implicit none
    save
    ! Equilibrium offset
    real(dp), dimension(numTens) :: deltas
    real(dp), allocatable :: J_Psi_ref(:,:,:), H_Psi_ref(:,:,:,:)

    ! Non-equilibrium offset
    real(dp), dimension(numMaxwell, numTens) :: deltas_a
    real(dp), allocatable :: Tau(:,:), J_tau(:,:,:,:), J_Psi_a_ref(:,:,:,:), H_Psi_a_ref(:,:,:,:,:), T_Psi_a_ref(:,:,:,:,:,:)
    allocate(J_Psi_ref(1,2,numTens)); J_Psi_ref = 0.0_dp
    allocate(H_Psi_ref(1,2,2,numTens)); H_Psi_ref = 0.0_dp

    allocate(Tau(numMaxwell,numTens)); Tau = 0.0_dp
    allocate(J_Tau(1,2,numMaxwell,numTens)); J_Tau = 0.0_dp
    allocate(J_Psi_a_ref(1,2,numMaxwell,numTens)); J_Psi_a_ref = 0.0_dp
    allocate(H_Psi_a_ref(1,2,2,numMaxwell,numTens)); H_Psi_a_ref = 0.0_dp
    allocate(T_Psi_a_ref(1,2,2,2,numMaxwell,numTens)); T_Psi_a_ref = 0.0_dp

    ! Stress offset for equilibrium free energy
    call vCANN_equi(invarsRef, J_Psi_ref, H_Psi_ref)
    deltas = J_Psi_ref(1,1,:) - J_Psi_ref(1,2,:)
    alphas = max(-deltas, 0.0_dp)
    betas  = max(deltas, 0.0_dp)
    deallocate(J_Psi_ref); deallocate(H_Psi_ref)

    ! Stress offset non-equilibrium free energy
    call vCANN_non_equi(invarsRef, invarsDotRef, Tau, J_Tau, J_Psi_a_ref, H_Psi_a_ref, T_Psi_a_ref)
    deltas_a = J_Psi_a_ref(1,1,:,:) - J_Psi_a_ref(1,2,:,:)
    alphas_a = max(-deltas_a, 0.0_dp)
    betas_a  = max(deltas_a, 0.0_dp)
    deallocate(Tau); deallocate(J_Tau); deallocate(J_Psi_a_ref); deallocate(H_Psi_a_ref); deallocate(T_Psi_a_ref)

  end subroutine init_offset

end module vCANNs

! ================================================================================================ !
! ===     Institute for Continuum and Material Mechanics, Hamburg University of Technology     === !
! ===                                                                                          === !
! ===                          Thermodynamically consistent vCANN UMAT                         === !
! ===                                                                                          === !
! ===                     Author: Kian Abdolazizi, kian.abdolazizi@tuhh.de                     === !
! ================================================================================================ 
    
subroutine UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD, &
RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN, &
TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,MATERL,NDI,NSHR,NTENS, &
NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT, &
DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

! Modules
! ================================================================
use precision
use parameters
use umat_statev_utils
use vCANNs

implicit none

! ================================================================
! ===                     DECLARATIONS                         ===
! ================================================================


! Formal variables
! ================================================================
character*8 :: MATERL

integer :: NDI,NSHR,NTENS,NSTATV,NPROPS,NOEL,NPT,LAYER,KSPT, &
    KSTEP,KINC

double precision :: STRESS,STATEV,DDSDDE,SSE,SPD,SCD,RPL,DDSDDT, &
    DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED, &
    PROPS,COORDS,DROT,PNEWDT,CELENT,DFGRD0,DFGRD1
    
dimension STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS), &
    DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS), &
    DFGRD0(3,3),DFGRD1(3,3),TIME(2),PREDEF(1),DPRED(1), &
    PROPS(NPROPS),COORDS(3),DROT(3,3)

! ================================================================
! Local variables
! ================================================================
integer :: aa, bb, ii, jj, mm, nn, kk, ll ! loop variables
integer, dimension(2,NTENS) :: indices    ! Voigt notation indices

! Equilibrium strain energy
! ================================================================
real(dp), allocatable :: J_Psi(:,:,:)      ! Jacobian of the equilibrium free energy w.r.t. the generalized invariants
real(dp), allocatable :: H_Psi(:,:,:,:)    ! Hessian of the equilibrium free energy w.r.t. the generalized invariants
real(dp) :: dU, ddU                        ! 1. and 2. derivative of the volumetric penalty function w.r.t. J

! non-equilibrium strain energy and relaxation times
! ================================================================
real(dp), allocatable :: J_Psi_a(:,:,:,:)      ! Jacobian of the non-equilibrium free energy w.r.t. the generalized invariants
real(dp), allocatable :: H_Psi_a(:,:,:,:,:)    ! Hessian of the non-equilibrium free energy w.r.t. the generalized invariants
real(dp), allocatable :: T_Psi_a(:,:,:,:,:,:)  ! third derivatives of the non-equilibrium free energy w.r.t. the generalized invariants

real(dp), allocatable :: tau(:,:)                                 ! relaxation times of current and previous increment
real(dp), allocatable :: J_Tau(:,:,:,:)                           ! Jacobians of the relaxation times w.r.t. the invariants
real(dp), dimension(numMaxwell,numTens) :: tau_bar, tau_old       ! average relaxation times during time increment

! Kinematics
! ================================================================
real(dp) :: det                                          ! determinant of the deforamtion gradient
real(dp), dimension(3,3) :: F, R, U                      ! deformation gradient, rotation tensor, right C.-G. deformation tensor
real(dp), dimension(3,3) :: Finv                         ! inverse deformation gradient
real(dp), dimension(numTens,3,3) :: L                    ! generalized structural tensors
real(dp), dimension(3,3) :: C                            ! right Cauchy-Green deformation tensor
real(dp), dimension(3,3) :: Cinv                         ! inverse of C
real(dp), dimension(3,3) :: Cbar_inv                     ! inverse of isochoric C
real(dp), dimension(numTens,3,3) :: Hbar                 ! C_bar_inv*L*C_bar_inv
real(dp), dimension(2*numTens) :: invars                 ! generalized invariants (I_1, J_1, I_2, J_2, ... )
real(dp), dimension(2*numTens+1) :: invars_dot           ! generalized invariants of \dot{Cbar}
real(dp), dimension(3,3,3,3) :: P_Dev                    ! Lagrangian deviatoric projection tensor

! Stresses of the generalized Maxwell models' branches
! ================================================================
real(dp), dimension(3,3,numTens) :: S_eq_r_bar             ! fictitious equilibrium 2. PK stresses
real(dp), dimension(3,3,numTens) :: S_eq_r                 ! deviatoric equilibrium 2. PK stressess
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_bar      ! fictitious auxiliary non-equilibrium 2. PK stresses
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_bar_old  ! old fictitious auxiliary non-equilibrium 2. PK stresses
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_neq_bar  ! fictirious non-equilbrium 2. PK stresses
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_neq      ! deviatoric non-equilbrium 2. PK stresses

! total stresses
! ================================================================
real(dp), dimension(3,3) :: S_eq     ! equilibrium 2. PK stress
real(dp), dimension(3,3) :: S_neq      ! non-equilibrium 2. PK stress
real(dp), dimension(3,3) :: S_eq_bar ! fictitious equilibrium 2. PK stress
real(dp), dimension(3,3) :: S_neq_bar  ! fictitious non-equilibrium 2. PK stress
real(dp), dimension(3,3) :: S_bar      ! fictitious 2. PK stress (eq. + non-eq.)
real(dp), dimension(3,3) :: S_iso      ! isochoric 2. PK stress
real(dp), dimension(3,3) :: S_vol      ! volumetric 2. PK stress
real(dp), dimension(3,3) :: S          ! total 2. PK stress
real(dp), dimension(3,3) :: cauchy     ! Cauchy stress
real(dp), dimension(3,3) :: kirchhoff  ! Cauchy stress

! Elasticity tensors of the generalized Maxwell models' equilibrium branchs
! ================================================================
real(dp), dimension(3,3,3,3,numTens) :: CC_eq_r_bar  ! fictitious elastic tangent moduli corresponding to the individual generalized Maxwell models
real(dp), dimension(3,3,3,3,numTens) :: CC_eq_r      ! actual isochoric equilibrium tangent moduli corresponding to the individual generalized Maxwell models

! Elasticity tensors of the generalized Maxwell models' non-equilibrium branchs
! ================================================================
real(dp), dimension(3,3,3,3,numMaxwell,numTens) :: CC_ra_bar
real(dp), dimension(3,3,3,3,numMaxwell,numTens) :: CC_ra_neq_bar
real(dp), dimension(3,3,3,3) :: CC_neq_bar

! total elasticity tensors
! ================================================================
real(dp), dimension(3,3,3,3) :: CC_eq_bar      ! equilibrium tangent
real(dp), dimension(3,3,3,3) :: CC_bar         ! total fictitious tangent
real(dp), dimension(3,3,3,3) :: CC_iso         ! isochoric tangent
real(dp), dimension(3,3,3,3) :: CC_vol         ! volumetric tangent
real(dp), dimension(3,3,3,3) :: CC_Jaumann     ! Jaumann tangent
real(dp), dimension(3,3,3,3) :: CC_Lagrangian  ! Lagrangian tangent
real(dp), dimension(3,3,3,3) :: CC_Eulerian    ! Eulerian tangent
real(dp), dimension(3,3,3,3) :: CC             ! total tangent
real(dp), dimension(6,6) :: CC_Jaumann_Voigt   ! Jaumann tangent in Voigt notation

! viscous overstresses 
! ================================================================
real(dp), dimension(3,3,numMaxwell,numTens) :: Q_ra      ! new viscous overstresses
real(dp), dimension(3,3,numMaxwell,numTens) :: Q_ra_old  ! old viscous overstresses
real(dp), dimension(3,3) :: Q                            ! total viscous overstress

! numerical tangent
! ================================================================
logical :: perturbation
real(dp), dimension(1,3) :: e_m, e_n           ! basis vectors for numerical perturbation
real(dp), dimension(3,3) :: dF                 ! increment of the deformation gradient due to numerical perturbation
real(dp), dimension(3,3,NTENS) :: kirchhoff_p  ! perturbed Cauchy stress
real(dp), dimension(3,3,NTENS) :: S_p          ! perturbed 2. PK stress
real(dp), dimension(NTENS,NTENS) :: CC_num     ! numerical Lagrangian elasticity tensor

real(dp), dimension(3) :: COORDS0


!if ((KINC == 1) .AND. (KSTEP == 1)) then
!    STATEV(57) = COORDS(1)
!    STATEV(58) = COORDS(2)
!    STATEV(59) = COORDS(3)
!end if
!COORDS0 = (/ STATEV(57), STATEV(58), STATEV(59) /)


! ================================================================
! ===                   GENERAL ASSIGNMENTS                    ===
! ================================================================

! Voigt indices
! ================================================================
if (NTENS .EQ. 6) then
	indices = reshape((/ 1, 1, 2, 2, 3, 3, 1, 2, 1, 3, 2, 3 /), shape(indices)) ! (2,6) 3d solid
else if (NTENS .EQ. 4) then
	indices = reshape((/ 1, 1, 2, 2, 3, 3, 1, 2/), shape(indices))              ! (2,4) 2d plane strain
else if (NTENS .EQ. 3) then
	indices = reshape((/ 1, 1, 2, 2, 1, 2/), shape(indices))                    ! (2,3) 2d plane stress
end if

! Deformation gradient
! ================================================================

! correction to deformation gradient for continuum elements in local coordinate system;
! ================================================================
! Fehervary et al. 2020 apply this correction only to 3D elements (ntens = 6),
! Terzano et al. 2023 apply this correction only to plane strain and 3d elements (ntens > 3) but not to plane stress elements (ntens=3).
! However, according to Nolan et. al 2020 this correction has also to be applied to plane stress elements too,
! but not to structural elements (membrane and shell elements). In essence, the correction has to be applied only
! if a local coordinate system and continuum elements (those whose identifier starts with a 'C') is used.

if ( (use_shell == .FALSE. ) .AND. (local_csys == .TRUE.) ) then
    call polardecomp_closed_form(DFGRD1,R,U)
    DFGRD1 = matmul(DFGRD1,transpose(R))
end if

! Explicitly enforce incompressibility for membrane and shell elements
! ================================================================
if (NTENS.EQ.3) then
	DFGRD1(3,3) = one/(DFGRD1(1,1)*DFGRD1(2,2)-DFGRD1(1,2)*DFGRD1(2,1))
endif

! F = DFGRD1

! ================================================================
! ===                     STATE VARIABLES                      ===
! ================================================================

call get_statevs(STATEV, S_ra_bar_old, Q_ra_old, tau_old, numMaxwell, numTens, NSTATV)

! ================================================================
! ===                    STRUCTURAL TENSORS                    ===
! ================================================================

call structural_tensors(L, COORDS0) ! compute the generalized structural tensors

! ================================================================
! ===             NUMERICAL TANGENT - PERTURBATIONS            ===
! ================================================================
perturbation = .FALSE.

if (use_numerical_tangent == .TRUE.) then
    perturbation = .TRUE.
    do kk = 1,NTENS
        
        ! basis vectors
	    ! ================================================================
        mm = indices(1,kk)
	    nn = indices(2,kk)
	    e_m = reshape((/ eye(mm,1), eye(mm,2), eye(mm,3) /), shape(e_m))
	    e_n = reshape((/ eye(nn,1), eye(nn,2), eye(nn,3) /), shape(e_n))
        
        
        ! perturbation for the Jaumann rate of the Kirchhoff stress --> ABAQUS tangent modulus
        ! ================================================================
        dF = epsilon/two*( matmul(transpose(e_m), matmul(e_n,DFGRD1)) + matmul(transpose(e_n), matmul(e_m,DFGRD1)) )
        
        if (use_shell == .TRUE.) then 
            ! perturbation for the Green-Naghdi rate of the Kirchhoff stress --> ABAQUS tangent modulus for shell elements
            ! ================================================================
            dF = epsilon/two*( matmul(matmul(transpose(e_m), e_n), F) + matmul(matmul(transpose(e_n), e_m), F) )
        end if

        ! perturbation for the 2. Piola-Kirchhoff stress tensors --> Lagrangian elasticity tensor
        ! ================================================================
        ! call inverse(DFGRD1, Finv)
        ! dF = epsilon/two*(matmul(matmul(transpose(Finv),transpose(e_m)), e_n) + matmul(matmul(transpose(Finv),transpose(e_n)), e_m))
        
        ! perturbed deformation gradient
        ! ================================================================
        F = DFGRD1 + dF
        
        ! correction for shell/membrane elements: F33 depends on the four independent
        ! components of the deformation gradient; any perturbation those components
        ! must cause a perturbation in F33
        if (NTENS .eq. 3) then
		    F(3,3) = one/(F(1,1)*F(2,2) - F(1,2)*F(2,1))  
	    end if  
        
        goto 10
20      continue
        
    end do
    
    perturbation = .FALSE.
    
end if

F = DFGRD1

! ================================================================
! ===                       KINEMATICS                         ===
! ================================================================
    
! 1.   deformation tensors, generalized invariants, ...
! ================================================================
! 10 call kinematics(F, L, C, Cinv, Cbar_inv, Hbar, invars, det)
10 call kinematics(F, DFGRD0, L, DTIME, C, Cinv, Cbar_inv, Hbar, invars, invars_dot, det)

! 2.   Lagranian deviatoric projection tensor
! ================================================================
call deviatoric_projection_tensor(eye, C, Cinv, P_Dev)

! ================================================================
! ===                         STRESS                           ===
! ================================================================

! ================================================================
! 1.    Evaluate neural networks
! ================================================================

! 1.1   Equilibrium free energy
! ================================================================
call vCANN_equi(invars, J_Psi, H_Psi) ! Jacobian and Hessian of equilibrium free energy

! 1.2   Non-equilibrium free energy
! ================================================================

! === 1.2.1   Referential relaxation time
if ((KINC == 1) .AND. (KSTEP == 1)) then
    call vCANN_non_equi(invarsRef, invars_dot, Tau, J_Tau, J_Psi_a, H_Psi_a, T_Psi_a)
    tau_old = tau
end if

! === 1.2.2   Current relaxation times and auxiliary nonequlibrium free energy
call vCANN_non_equi(invars, invars_dot, Tau, J_Tau, J_Psi_a, H_Psi_a, T_Psi_a) 

! === 1.2.3   Average relaxation times
tau_bar = (tau + tau_old)/two


! ================================================================
! 2.    Compute isochoric stresses
! ================================================================

! 2.1   Fictitious and deviatoric 2. PK stress tensors
! ================================================================
call equilibrium_and_auxiliary_stress(det, invars, J_Psi(1,1:2,:), &
    J_Psi_a(1,1:2,:,:), L, Cbar_inv, Hbar, S_eq_r_bar, S_eq_r, S_ra_bar)

! 2.2   Update auxiliary non-equilibrium stresses Q_ra
! ================================================================
call auxiliary_stress_update(S_ra_bar, S_ra_bar_old, Q_ra_old, &
    DTIME, tau_bar, Q_ra)

! 2.2   Compute non-equilibrium stresses S_ra_neq
! ================================================================
call non_equilibrium_stress(Q_ra, J_Psi_a(1,1:2,:,:), H_Psi_a(1,1:2,1:2,:,:), & 
    L, Hbar, det, C, Cinv, Cbar_inv, S_ra_neq, S_ra_neq_bar)


! ================================================================
! 3.   Volumetric equilibrium contribution
! ================================================================
dU = kappa*(det - one/det) ! first derivative of the volumetric penalty function U(J), dU/dJ
ddU = kappa*(1 + one/det**two) ! second derivative of the volumetric penalty function U(J), d^2U/dJ^2
call volumetric_2_PK_stress(det, dU, Cinv, S_vol)


! ================================================================
! 4.   Total stress
! ================================================================

S_eq = sum(S_eq_r, dim=3)              ! equilibirum stress
S_neq  = sum(sum(S_ra_neq, dim=4),dim=3)   ! non-equilibrium stress
S_iso = S_eq + S_neq                     ! total isochoric stress

S_eq_bar = sum(S_eq_r_bar, dim=3)             ! fictitious equilibrium stress
S_neq_bar  = sum(sum(S_ra_neq_bar, dim=4),dim=3)  ! fictitious non-equilibrium stress
S_bar = S_eq_bar + S_neq_bar                    ! total fictitious stress

! 4.2   Add all contributions
! ================================================================

! 4.3   Apply incompressibility constraint (for debugging in Visual Studio and for shell and plane stress elements in ABAQUS )
! ================================================================
S = S_iso ! - S_iso(3,3)/Cinv(3,3)*Cinv

! 4.4   Push-forward to Cauchy stress tensor
! ================================================================
call cauchy_stress(S, F, det, cauchy)

! 4.5   Kirchhoff stress tensor
! ================================================================
kirchhoff = det*cauchy

! ================================================================
! ===             NUMERICAL TANGENT - CALCULATION              ===
! ================================================================

if (perturbation == .TRUE.) then
    
    ! perturbed Kirchhoff stresses for ABAQUS tangent
    kirchhoff_p(:,:,kk) = kirchhoff
    
    ! perturbed 2. PK stresses for Lagrangian elasticity tensor
    S_p(:,:,kk) = S_iso
    
    deallocate(J_Psi)
    deallocate(H_Psi)
    deallocate(Tau)
    deallocate(J_Tau)
    deallocate(J_Psi_a)
    deallocate(H_Psi_a)
    deallocate(T_Psi_a)
    
    goto 20
    
end if

if (use_numerical_tangent == .TRUE.) then
    do kk = 1,NTENS
        do ll = 1,NTENS
            aa = indices(1,ll)
            bb = indices(2,ll)
            
            ! numerical ABAQUS tangent
            DDSDDE(ll,kk) = one/(det*epsilon)*(kirchhoff_p(aa,bb,kk) - kirchhoff(aa,bb))
            
            ! numerical Langrangian elasticity tensor
            CC_num(ll,kk) = one/epsilon*(S_p(aa,bb,kk) - S_iso(aa,bb))
        end do
    end do
    call voigt_notation_stress(cauchy, NTENS, indices, STRESS) ! convert Cauchy stress in to Voigt notation 
end if

! ================================================================
! ===                     STATE VARIABLES                      ===
! ================================================================

call set_statevs(S_ra_bar, Q_ra, tau, STATEV, numMaxwell, numTens, NSTATV)
    
! ================================================================
! ===                    ANALYTICAL TANGENT                    ===
! ================================================================

if (use_numerical_tangent == .FALSE.) then

    ! 1.   Equilibrium and auxiliary non-equilbrium contribution
    ! ================================================================

    ! 1.1   Fictitious tangent moduli
    ! ================================================================
    call fictitious_tangent_moduli(J_Psi(1,1:2,:), H_Psi(1,1:2,1:2,:), &
        J_Psi_a(1,1:2,:,:), H_Psi_a(1,1:2,1:2,:,:), Hbar, Cbar_inv, L, &
        CC_eq_r_bar, CC_ra_bar)

    CC_eq_bar = sum(CC_eq_r_bar, dim=5)

    call fictitious_nonequilibrium_tangent(J_Psi_a(1,1:2,:,:), &
        H_Psi_a(1,1:2,1:2,:,:), T_Psi_a(1,1:2,1:2,1:2,:,:), L, &
        Hbar, Cbar_inv, Q_ra, tau_bar, DTIME, S_ra_bar, Q_ra_old, &
        S_ra_bar_old, J_Tau(1,1:2,:,:), CC_ra_bar, CC_ra_neq_bar)

    CC_neq_bar = sum(sum(CC_ra_neq_bar, dim=6), dim=5)
    
    CC_bar = CC_eq_bar + CC_neq_bar
    
    ! 1.2   Actual tangent moduli
    ! ================================================================
    call lagrangian_isochoric_tangent(CC_bar, P_Dev, S_iso, C, Cinv, &
        S_bar, det, CC_iso)

    ! 2.   Volumetric contribution
    ! ================================================================
    call lagrangian_volumetric_tangent(dU, ddU, Cinv, det, CC_vol)

    ! 4.   Jaumann rate contribution
    ! ================================================================
    call jaumann_tangent(cauchy, eye, CC_Jaumann) ! if plane strain/stress elements are used, this is the way
    ! call jaumann_tangent_voigt(cauchy, CC_Jaumann_Voigt) ! directly in Voigt notation only for continuum elements right now (if I am correct), has to be modified for plane strain/stress elements due to other 'indices'

    ! 5.   Total tangent modulus
    ! ================================================================
    CC_Lagrangian = CC_iso + CC_vol                             ! Lagrangian tangent modulus
    call eulerian_tangent(CC_Lagrangian, F, det, CC_Eulerian)   ! Push-forward to Eulerian tangent modulus
    CC = CC_Eulerian + CC_Jaumann                      ! Add correction for ABAQUS' tangent formulation
    
    ! 6.   Convert stress and tangent to Voigt notation; correct for the Jaumann rate formulation of ABAQUS
    ! ================================================================
    call voigt_notation_stress(cauchy, NTENS, indices, STRESS)    
    call voigt_notation_tangent(CC, NTENS, indices, DDSDDE)
    ! DDSDDE = DDSDDE + CC_Jaumann_Voigt
    
end if

deallocate(J_Psi)
deallocate(H_Psi)
deallocate(Tau)
deallocate(J_Tau)
deallocate(J_Psi_a)
deallocate(H_Psi_a)
deallocate(T_Psi_a)

!! Deformation gradient
!STATEV(48) = DFGRD1(1,1)
!STATEV(49) = DFGRD1(1,2)
!STATEV(50) = DFGRD1(1,3)
!STATEV(51) = DFGRD1(2,1)
!STATEV(52) = DFGRD1(2,2)
!STATEV(53) = DFGRD1(2,3)
!STATEV(54) = DFGRD1(3,1)
!STATEV(55) = DFGRD1(3,2)
!STATEV(56) = DFGRD1(3,3)

end subroutine UMAT

! ================================================================
! ===                        END UMAT                          ===
! ================================================================

    
    
! ================================================================================================ !  
! ===                                      Strucutral tensors                                  === !
! ================================================================================================ !      
subroutine structural_tensors(L, COORDS)

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(numTens,3,3), intent(out) :: L    ! generalized structural tensors
    real(dp), dimension(3), intent(in) :: COORDS          ! the coordinates

    ! internal varibales
    ! ================================================================
    real(dp), dimension(numDir,3) :: material_dirs        ! preferred material directions
    real(dp), dimension(3) :: dir                         ! preferred material directions
    real(dp), dimension(numTens,numDir+1) :: dir_weights  ! scalar weighting factors of the generalized structural tensors
    real(dp), dimension(3,3) :: L_0                       ! isotropic structural tensor
    real(dp), dimension(numDir,3,3) :: L_rj               ! structural tensors
    integer :: jj, ii
    real(dp) :: alpha                             ! polar angle 


    L_0 = eye/three ! isotropic contribution

    if (numDir == 0) then
        do jj = 1,numTens
            L(jj,:,:) = L_0
        end do
    else
        ! preferred material directions and scalar weights (should be calculated in the beginning)
        ! ================================================================
        call vCANN_structural(extra_struc, material_dirs, dir_weights)
    
        ! if a global csys is used instead of a local one, compute the fiber vectors in the initial
        ! configuration wrt. the global csys (here only for cylindircal coordinates)
        ! ================================================================
        if (local_csys == .FALSE.) then
            ! compute the polar angle
            alpha = atan2(COORDS(1), COORDS(3))
            do jj = 1,numDir/2
                ! compute the fiber vector by revolving a generating fiber vector around the cylinder
                material_dirs(2*jj-1,:) = (/ cos(alpha)*cos(theta(jj)),  sin(theta(jj)), -sin(alpha)*cos(theta(jj)) /)
                material_dirs(2*jj,:)   = (/ cos(alpha)*cos(theta(jj)), -sin(theta(jj)), -sin(alpha)*cos(theta(jj)) /)
            end do        
        end if
    

        ! structural tenors
        ! ================================================================
        do ii = 1,numDir
            dir = material_dirs(ii,:)
            L_rj(ii,:,:) = spread(dir,dim=2,ncopies=3) * spread(dir,dim=1,ncopies=3)
        end do

        ! generalized structural tensors
        ! ================================================================
        L = zero
        do jj = 1,numTens
            L(jj,:,:) = L_0*dir_weights(jj,1)
            do ii = 1,numDir
                L(jj,:,:) = L(jj,:,:) + L_rj(ii,:,:)*dir_weights(jj,ii+1)
            end do
        end do
    end if

end subroutine structural_tensors
    
    
     
! ================================================================================================ !  
! ===                                          Kinematics                                      === !
! ================================================================================================ !  
subroutine kinematics(F1, F0, L, DTIME, C, Cinv, Cbar_inv, Hbar, invars, invars_dot, det)

    ! Computes the deformation tensors and generalized invariants as well as
    ! tensors bases for stress and elasticity tensor

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: F1, F0                        ! deformation gradients
    real(dp), dimension(numTens,3,3), intent(in) :: L                     ! generalized structural tensors
    real(dp), intent(in) :: DTIME                                         ! time increment

    real(dp), dimension(3,3), intent(out) :: C                            ! right Cauchy-Green deformation tensor
    real(dp), dimension(3,3), intent(out) :: Cinv                         ! inverse of C
    real(dp), dimension(3,3), intent(out) :: Cbar_inv                     ! inverse of isochoric C
    real(dp), dimension(numTens,3,3), intent(out) :: Hbar                 ! C_bar_inv*L*C_bar_inv
    real(dp), dimension(2*numTens), intent(out) :: invars                 ! generalized invariants (I_1, J_1, I_2, J_2, ... )
    real(dp), dimension(2*numTens+1), intent(out) :: invars_dot           ! generalized invariants \dot{Cbar}
    real(dp), intent(out) :: det                                          ! J, determinant of the deforamtion gradient

    ! internal variables
    ! ================================================================
    real(dp), dimension(3,3) :: Cbar        ! isochoric right Cauchy-Green deformation tensor
    real(dp), dimension(3,3) :: cofC        ! cofactor of C
    real(dp), dimension(3,3) :: cofCbar     ! cofactor of isochoric C
    real(dp), dimension(3,3) :: CbarL       ! matrix product of C and a generalized structural tensor
    real(dp), dimension(3,3) :: cofCbarL    ! matrix product of cofactor of C and a generalized structural tensor
    real(dp), dimension(3,3) :: Fdot        ! material time derivative of F
    real(dp), dimension(3,3) :: Cdot        ! material time derivative of C
    real(dp), dimension(3,3) :: Cbardot     ! material time derivative of Cbar
    real(dp), dimension(3,3) :: cofCbardot  !cofactor of material time derivative of Cbar
    real(dp), dimension(3,3) :: CbardotL, cofCbardotL
    real(dp) :: trCinvCdot, detCbardot

    integer :: ii ! loop variables

    ! ================================================================
    ! Compute the kinemtics
    ! ================================================================

    call determinant(F1, det) ! Jacobian determinant J
    
    ! deformation measures
    ! ================================================================
    C = matmul(transpose(F1),F1)    ! right Cauchy-Green deformation tensor
    Cbar = det**(-two/three) * C  ! isochroic right Cauchy-Green deformation tensor
    
    call cofactor(C, cofC)        ! cofactor of C
    Cinv = cofC/det**two          ! inverse of C

    call cofactor(Cbar, cofCbar)  ! cofactor of Cbar
    Cbar_inv = cofCbar            ! inverse of Cbar
    
    if (rateDependent == .TRUE.) then
      call compute_Fdot(F0, F1, DTIME, Fdot)
      call compute_Cdot(F1, Fdot, Cdot)

      call double_contraction_22(Cinv, Cdot, trCinvCdot)
      Cbardot = det**(-two/three)*(Cdot - trCinvCdot/three*C)
      call cofactor(Cbardot, cofCbardot)        ! cofactor of C
    end if

    do ii = 1,numTens
        Hbar(ii,:,:) = matmul(Cbar_inv, matmul(L(ii,:,:), Cbar_inv)) ! Cinv*L_tilde*Cinv
    end do

    ! generalized isochoric invariants
    ! ================================================================
    do ii = 1,numTens
        ! of the isochoric right Cauchy deformation gradient
        CbarL = matmul(Cbar, L(ii,:,:))
        cofCbarL = matmul(cofCbar, L(ii,:,:))  
    
        invars(2*ii-1) = CbarL(1,1) + CbarL(2,2) + CbarL(3,3)           ! first invariant 
        invars(2*ii)   = cofCbarL(1,1) + cofCbarL(2,2) + cofCbarL(3,3)  ! second invariant
        
        ! of the material time derivative of the isochoric right Cauchy deformation gradient
        if (rateDependent == .TRUE.) then
          CbardotL = matmul(Cbardot, L(ii,:,:))
          cofCbardotL = matmul(cofCbardot, L(ii,:,:))  
    
          invars_dot(2*ii-1) = CbardotL(1,1) + CbardotL(2,2) + CbardotL(3,3)           ! first invariant 
          invars_dot(2*ii)   = cofCbardotL(1,1) + cofCbardotL(2,2) + cofCbardotL(3,3)  ! second invariant
        end if
    
    end do
    
    call determinant(Cbardot, detCbardot)
    invars_dot(2*numTens+1) = detCbardot

end subroutine kinematics
    
! ================================================================================================ !  
! ===                               Fictitious and deviatoric stress                           === !
! ================================================================================================ !  
subroutine equilibrium_and_auxiliary_stress(det, invars, J_Psi, J_Psi_a, L, Cbar_inv, Hbar, S_eq_bar, S_eq_dev, S_ra_bar)

    ! computes the fictitious and deviatoric 2. Piola-Kirchhoff stress tensor

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: det 
    real(dp), dimension(2*numTens), intent(in) :: invars                    ! generalized invariants (I_1, J_1, I_2, J_2, ... )
    real(dp), dimension(1,2,numTens), intent(in) :: J_Psi                   ! Jacobian of the equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,numMaxwell,numTens), intent(in) :: J_Psi_a      ! Jacobian of the auxiliary non-equilibrium free energy w.r.t. the generalized invariants    real(dp), dimension(numTens,3,3), intent(in) :: L                       ! generalized structural tensors
    real(dp), dimension(numTens,3,3), intent(in) :: L                       ! generalized structural tensors
    real(dp), dimension(3,3), intent(in) :: Cbar_inv                        ! inverse of the isochoric right Cauchy-Green deformation tensor
    real(dp), dimension(numTens,3,3), intent(in) :: Hbar                    ! C_bar_inv*L*C_bar_inv
    real(dp), dimension(3,3,numTens), intent(out)  :: S_eq_bar              ! fictitious equilibrium 2. PK stress tensors
    real(dp), dimension(3,3,numTens), intent(out)  :: S_eq_dev              ! deviatoric equilibrium 2. PK stresses
    real(dp), dimension(3,3,numMaxwell,numTens), intent(out)  :: S_ra_bar   ! fictitious auxiliary stress 2. PK stress tensors


    ! internal variables
    ! ================================================================
    integer :: ii, jj ! loop variables


    do ii = 1,numTens
        
        ! ficticious 2. PK stress tensor S_eq_bar
        S_eq_bar(:,:,ii) = two*(J_Psi(1,1,ii)*L(ii,:,:) - J_Psi(1,2,ii)* Hbar(ii,:,:)) 
    
        ! deviotoric  2. PK stress tensor (deviatoric projection of fictitious stress)
        S_eq_dev(:,:,ii) = det**(-two/three)*two*( J_Psi(1,1,ii)*(L(ii,:,:) - invars(2*ii-1)/three * Cbar_inv)  &
                           - J_Psi(1,2,ii)*(Hbar(ii,:,:) - invars(2*ii)/three * Cbar_inv) )

        ! ficticious 2. PK stress tensor S_ra_bar (deviatoric equivalent is never used)
        do jj = 1,numMaxwell
            S_ra_bar(:,:,jj,ii) = two*(J_Psi_a(1,1,jj,ii)*L(ii,:,:) - J_Psi_a(1,2,jj,ii)* Hbar(ii,:,:))
            
        end do
    end do

    end subroutine equilibrium_and_auxiliary_stress


! ================================================================================================ !  
! ===                                        Viscous Stress                                    === !
! ================================================================================================ !   
subroutine auxiliary_stress_update(S_ra_bar, S_ra_bar_old, Q_ra_old, DTIME, tau_bar, Q_ra)

    ! update the auxiliary non-equilibrium stresses using the recurrence update formula

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: S_ra_bar      ! instantaneous elastic 2. PK stress
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: S_ra_bar_old  ! old ficticious elastic stress
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: Q_ra_old      ! old auxiliarystresses
    real(dp), intent(in) :: DTIME                                            ! time increment
    real(dp), dimension(numMaxwell,numTens), intent(in) :: tau_bar           ! average relaxation times during time increment

    real(dp), dimension(3,3,numMaxwell,numTens), intent(out) :: Q_ra         ! new  auxiliary stresses

    ! internal variables
    ! ================================================================
    integer :: ii, jj ! loop variables
    real(dp) :: xi

    ! update viscous overstresses
    ! ================================================================
    do ii = 1,numTens
        do jj = 1,numMaxwell
            xi = -DTIME/(two*tau_bar(jj,ii))
            Q_ra(:,:,jj,ii) = exp(xi)*S_ra_bar(:,:,jj,ii) &
                            + exp(xi)*(exp(xi)*Q_ra_old(:,:,jj,ii) - S_ra_bar_old(:,:,jj,ii))
        end do
    end do

end subroutine auxiliary_stress_update
    
    
! ================================================================================================ !  
! ===                                     Non-equilibrium stress                               === !
! ================================================================================================ !      
subroutine non_equilibrium_stress(Q_ra, J_Psi_a, H_Psi_a, L, Hbar, det, C, Cinv, Cbar_inv, S_ra_neq_dev, S_ra_neq_bar)

    ! Computes the non-equilibrium stress using the explicit forumal

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: Q_ra          ! auxiliary stresses
    real(dp), dimension(1,2,numMaxwell,numTens), intent(in) :: J_Psi_a       ! Jacobian of the auxiliary non-equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,2,numMaxwell,numTens), intent(in) :: H_Psi_a     ! Hessian of the auxiliary non-equilibrium free energy w.r.t. the generalized invariants 
    real(dp), dimension(numTens,3,3), intent(in) :: L                        ! generalized structural tensors
    real(dp), dimension(3,3), intent(in) :: C, Cinv, Cbar_inv                ! [inverse of the (isochoric)] right Cauchy-Green deformation tensor
    real(dp), intent(in) :: det
    real(dp), dimension(numTens,3,3), intent(in) :: Hbar                     ! C_bar_inv*L*C_bar_inv
    real(dp), dimension(3,3,numMaxwell,numTens), intent(out) :: S_ra_neq_bar ! fictitious non-equilibrium stress
    real(dp), dimension(3,3,numMaxwell,numTens), intent(out) :: S_ra_neq_dev ! deviatoric non-equilibrium stress

    ! internal variables
    ! ================================================================
    integer :: ii, jj ! loop variables
    real(dp), dimension(3,3) :: term_1, term_2, term_3, term_4, M, S_ra_neq_barC
    real(dp) :: eps, gamma, trS_ra_neq_bar
    
    do ii = 1,numTens
        do jj = 1,numMaxwell
            
            call double_contraction_22(Hbar(ii,:,:), Q_ra(:,:,jj,ii), gamma)
            call double_contraction_22(L(ii,:,:), Q_ra(:,:,jj,ii), eps)
            M = matmul(matmul(Cbar_inv, Q_ra(:,:,jj,ii)),Hbar(ii,:,:))
            
            term_1 = H_Psi_a(1,1,1,jj,ii)*eps*L(ii,:,:)
            term_2 = H_Psi_a(1,2,2,jj,ii)*gamma*Hbar(ii,:,:)
            term_3 = J_Psi_a(1,2,jj,ii)*( M + transpose(M))
            term_4 = H_Psi_a(1,1,2,jj,ii)*(gamma*L(ii,:,:) + eps*Hbar(ii,:,:))
            
            S_ra_neq_bar(:,:,jj,ii) = two*(term_1 + term_2 + term_3 - term_4)
            
            S_ra_neq_barC = matmul(S_ra_neq_bar(:,:,jj,ii), C)
            trS_ra_neq_bar = S_ra_neq_barC(1,1) + S_ra_neq_barC(2,2) + S_ra_neq_barC(3,3)
            S_ra_neq_dev(:,:,jj,ii) = det**(-two/three)*(S_ra_neq_bar(:,:,jj,ii) - trS_ra_neq_bar/three*Cinv)
            
        end do
    end do
    
end subroutine non_equilibrium_stress
    

! ================================================================================================ !  
! ===                                       Volumteric Stress                                  === !
! ================================================================================================ !   
subroutine volumetric_2_PK_stress(det, dU, Cinv, S_vol)

    ! Computes the volumetric stress contribution; dispensable for incompressible materials

    use precision
    implicit none
    
    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: det                     ! determination of the deformation gradient
    real(dp), intent(in) :: dU                      ! 1. derivative of the volumetric penalty function w.r.t. J
    real(dp), dimension(3,3), intent(in) :: Cinv    ! inverse of C

    real(dp), dimension(3,3), intent(out) :: S_vol  ! volumetric 2. PK stress

    S_vol = det*dU*Cinv ! volumetric 2. Piola Kirchhoff stress tensor S_vol

end subroutine volumetric_2_PK_stress    
    

! ================================================================================================ !  
! ===                                 Fictitious tangent moduli                                === !
! ================================================================================================ !  
subroutine fictitious_tangent_moduli(J_Psi, H_Psi, J_Psi_a, H_Psi_a,Hbar, Cbar_inv, L, CC_eq_r_bar, CC_ra_bar)

    ! Computes the fictitious equlibrium and auxialiary non-equlibrium tangent moduli 
    ! CC_eq_r_bar = 2* dS_eq_r_bar/dC_bar and CC_ra_bar = 2* dS_ra_bar/dC_bar, respectively,
    ! S_eq_r_bar is the ficitious equilibrium 2. PK stress and S_ra_bar is the fictitious
    ! auxiliary non-equlibrium 2. PK stress. C_bar is the isochoric right Cauchy deformation tensor.

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(1,2,numTens), intent(in) :: J_Psi                      ! Jacobian of the equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,2,numTens), intent(in) :: H_Psi                    ! Hessian of the equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,numMaxwell,numTens), intent(in) :: J_Psi_a         ! Jacobian of the auxiliary non-equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,2,numMaxwell,numTens), intent(in) :: H_Psi_a       ! Hessian of the auxiliary non-equilibrium free energy w.r.t. the generalized invariants
    real(dp), dimension(numTens,3,3), intent(in) :: Hbar                               ! C_bar_inv*L*C_bar_inv
    real(dp), dimension(3,3), intent(in) :: Cbar_inv                           ! isochoric inverse of right Cauchy-Green deformation tensor
    real(dp), dimension(numTens,3,3), intent(in) :: L                          ! generalized strucutral tensor

    real(dp), dimension(3,3,3,3,numTens), intent(out) :: CC_eq_r_bar         ! fictitious equilibrium tangent modulus
    real(dp), dimension(3,3,3,3,numMaxwell,numTens), intent(out) :: CC_ra_bar  ! fictitious auxiliary non-equilibrium tangent modulus

    ! internal variables
    ! ================================================================
    integer :: ii, jj
    real(dp), dimension(3,3,3,3) :: L_odot_L, Hbar_odot_Hbar, &
        Cbar_inv_otimes_Hbar_sym, Hbar_otimes_Cbar_inv_sym, L_odot_Hbar, &
        Hbar_odot_L, N, M

    do ii = 1,numTens

        ! compute intermediate variables
        call A_odot_B(L(ii,:,:), L(ii,:,:), L_odot_L)
        call A_odot_B(Hbar(ii,:,:), Hbar(ii,:,:), Hbar_odot_Hbar)
        call A_otimes_B_sym(Cbar_inv, Hbar(ii,:,:), Cbar_inv_otimes_Hbar_sym)
        call A_otimes_B_sym(Hbar(ii,:,:), Cbar_inv, Hbar_otimes_Cbar_inv_sym)
        call A_odot_B(L(ii,:,:), Hbar(ii,:,:), L_odot_Hbar)
        call A_odot_B(Hbar(ii,:,:), L(ii,:,:), Hbar_odot_L)
        N = Cbar_inv_otimes_Hbar_sym + Hbar_otimes_Cbar_inv_sym
        M = L_odot_Hbar + Hbar_odot_L
        
        ! fictitious equilibrium tangent moduli
        CC_eq_r_bar(:,:,:,:,ii) = four*( H_Psi(1,1,1,ii)*L_odot_L   &
                      + H_Psi(1,2,2,ii)*Hbar_odot_Hbar   &
                      + J_Psi(1,2,ii)*N &
                      - H_Psi(1,1,2,ii)*M )

        ! fictitious auxiliary non-equilibrium tangent moduli
        do jj = 1,numMaxwell
            CC_ra_bar(:,:,:,:,jj,ii) = four*( H_Psi_a(1,1,1,jj,ii)*L_odot_L   &
                          + H_Psi_a(1,2,2,jj,ii)*Hbar_odot_Hbar   &
                          + J_Psi_a(1,2,jj,ii)*N &
                          - H_Psi_a(1,1,2,jj,ii)*M )
        end do
    end do
    

end subroutine fictitious_tangent_moduli

    
! ================================================================================================ !  
! ===                Lagrangian actual instantaneous elastic tangent modulus                   === !
! ================================================================================================ !  
subroutine lagrangian_isochoric_tangent(CC_eq_r_bar, P_Dev, S_iso, C, Cinv, S_bar, J, CC_eq_r)

    ! Computes the actual instantaneous elastic tangent modulus

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: J                                  ! Jacobian determinant, det(F)
    real(dp), dimension(3,3), intent(in) :: S_iso              ! isochoric 2.PK stress
    real(dp), dimension(3,3), intent(in) :: S_bar              ! ficticious 2.PK stress
    real(dp), dimension(3,3), intent(in) :: C, Cinv           ! right Cauchy-Green deformation tensor and its inverse
    real(dp), dimension(3,3,3,3), intent(in) :: P_Dev          ! Lagrangian deviatoric projection tensor
    real(dp), dimension(3,3,3,3), intent(in) :: CC_eq_r_bar  ! ficticious equilibrium tangent modulus

    real(dp), dimension(3,3,3,3), intent(out) :: CC_eq_r     ! actual equilibrium tangent modulus

    ! internal auxiliary variables
    ! ================================================================
    real(dp) :: C_double_dot_S_bar
    real(dp), dimension(3,3,3,3) :: P_Dev_T, S_iso_odot_Cinv, Cinv_odot_S_iso,  &
        Cinv_otimes_Cinv_sym, Cinv_odot_Cinv, P_Dev_CC_eq_r_bar_P_Dev_T, term_1, term_2, term_3

    ! calculate intermediate / auxiliary variables
    ! ================================================================
    call transpose_fourth_order(P_Dev, P_Dev_T)
    call A_odot_B(S_iso, Cinv, S_iso_odot_Cinv)
    call A_odot_B(Cinv, S_iso, Cinv_odot_S_iso)
    call A_otimes_B_sym(Cinv, Cinv, Cinv_otimes_Cinv_sym)
    call A_odot_B(Cinv, Cinv, Cinv_odot_Cinv)
    call double_contraction_22(C, S_bar, C_double_dot_S_bar)
    call composition_fourth_order_ABC(P_Dev, CC_eq_r_bar, P_Dev_T, P_Dev_CC_eq_r_bar_P_Dev_T) ! deviatoric projection of CC_eq_r_bar

    ! finally, calculate isochoric elastic tangent
    ! ================================================================
    term_1 = J**(-four/three)*P_Dev_CC_eq_r_bar_P_Dev_T
    term_2 = - two/three*(S_iso_odot_Cinv + Cinv_odot_S_iso)
    term_3 = two/three*J**(-two/three)*C_double_dot_S_bar*(Cinv_otimes_Cinv_sym - one/three*Cinv_odot_Cinv)

    CC_eq_r = term_1 + term_2 + term_3

end subroutine lagrangian_isochoric_tangent
    
    
! ================================================================================================ !  
! ===                         Lagrangian volumetric elastic tangent modulus                    === !
! ================================================================================================ !   
subroutine lagrangian_volumetric_tangent(dU, ddU, Cinv, J, CC_vol)

    ! Computes the volumetric elastic tangent modulus

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: J                           ! Jacobian determinant
    real(dp), intent(in) :: dU, ddU                     ! 1. and 2. derivative of the volumetric free energy w.r.t. J
    real(dp), dimension(3,3), intent(in) :: Cinv       ! inverse of the right Cauchy-Green deformation tensor

    real(dp), dimension(3,3,3,3), intent(out) :: CC_vol ! volumeric tangent modulus

    ! internal auxiliary variables
    ! ================================================================
    real(dp), dimension(3,3,3,3) :: Cinv_otimes_Cinv_sym, Cinv_odot_Cinv
    
    ! calculate intermediate / auxiliary variables
    ! ================================================================
    call A_otimes_B_sym(Cinv, Cinv, Cinv_otimes_Cinv_sym)
    call A_odot_B(Cinv, Cinv, Cinv_odot_Cinv)

    ! finally, calculate volumetric elastic tangent
    ! ================================================================
    CC_vol = (J**two*ddU + J*dU)*Cinv_odot_Cinv - two*J*dU*Cinv_otimes_Cinv_sym
 
end subroutine lagrangian_volumetric_tangent
    
    
! ================================================================================================ !  
! ===                                    Eulerian tangent modulus                              === !
! ================================================================================================ !   
subroutine eulerian_tangent(CC_Lagrangian, F, J, CC_Eulerian)
  
    ! Computes the Eulerian / spatial tangent modulus by a push-forward / Piola transformation of the
    ! Lagrangian / material elasticity tensor
    
    use precision
    implicit none
    
    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: J                       ! Jacobian determinant
    real(dp), intent(in) :: F(3,3)                  ! deformation gradient
    real(dp), intent(in) :: CC_Lagrangian(3,3,3,3)  ! Lagrangian tangent modulus
    real(dp), intent(out) :: CC_Eulerian(3,3,3,3)   ! Eularian tangent modulus

    ! internal variables (9x9 reindexed view)
    ! ================================================================
    real(dp) :: CL9(9,9), M(9,9), T(9,9), CE9(9,9)
    integer :: ii,jj,kk,ll, I,Jc

    ! Build M = F ⊗ F  with I=i+3(j-1), J=k+3(l-1)
    do ll = 1,3
        do kk = 1,3
            Jc = kk + 3*(ll-1)
            do jj = 1,3
                do ii = 1,3
                    I  = ii + 3*(jj-1)
                    M(I,Jc) = F(ii,kk) * F(jj,ll)
                end do
            end do
        end do
    end do

    ! Reshape material tensor to 9x9 (rows: m,n ; cols: o,p)
    CL9 = reshape(CC_Lagrangian, (/9,9/))

    ! Push-forward: CE9 = (1/J) * M * CL9 * M^T
    T   = matmul(CL9, transpose(M))   ! right multiply by M^T
    CE9 = matmul(M, T)                ! left multiply by M
    CE9 = CE9 / J

    ! Back to (i,j,k,l)
    CC_Eulerian = reshape(CE9, (/3,3,3,3/))
    
end subroutine eulerian_tangent

    
! ================================================================================================ !  
! ===                                      Cauchy stress tensor                                === !
! ================================================================================================ !  
subroutine cauchy_stress(S, F, J, cauchy)

    ! Computes the Cauchy stress tensor by push-forward / Piola transformationn of the 2. PK stress
    ! sigma = 1/J*F*S*F^T

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in) :: J                       ! Jacobian determinant
    real(dp), dimension(3,3), intent(in) :: F       ! deformation gradient
    real(dp), dimension(3,3), intent(in) :: S       ! 2. PK stress tensor

    real(dp), dimension(3,3), intent(out) :: cauchy ! Cauchy stress tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll
    real(dp) :: tmp

    do ii = 1,3
        do jj = 1,3
            tmp = 0.d0
            do kk = 1,3
                do ll = 1,3
                    tmp = tmp + F(ii,kk)*S(kk,ll)*F(jj,ll)
                end do
            end do
            cauchy(ii,jj) = tmp/J
        end do
    end do

end subroutine cauchy_stress
    
    
! ================================================================================================ !  
! ===        Tangent modulus contribution due to Jaumann rate of Kirchhoff stress in ABAQUS    === !
! ================================================================================================ !            
subroutine jaumann_tangent(cauchy, eye, CC_Jaumann)

    ! Computes the contribution of the tangent modulus which is attributed to the Jaumann rate of the 
    ! Kirchhoff stress in ABAQUS. Possesses minor and major symmtries. Independent of the constitutive
    ! model.

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: cauchy           ! Cauchy stress tensor
    real(dp), dimension(3,3), intent(in) :: eye              ! second-order identity tensor

    real(dp), dimension(3,3,3,3), intent(out) :: CC_Jaumann  ! Jaumann contribution to the tangent modulus

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ii = 1,3
        do jj = 1,3
            do kk = 1,3
                do ll = 1,3
                    CC_Jaumann(ii,jj,kk,ll) = ( eye(ii,kk)*cauchy(jj,ll)   &
                                              + eye(ii,ll)*cauchy(jj,kk)   &
                                              + eye(jj,kk)*cauchy(ii,ll)   &
                                              + eye(jj,ll)*cauchy(ii,kk) )/2.d0
                end do
            end do
        end do
    end do

end subroutine jaumann_tangent

    
! ================================================================================================ !  
! ===        Tangent modulus contribution due to Jaumann rate of Kirchhoff stress in ABAQUS    === !
! ================================================================================================ !            
subroutine jaumann_tangent_voigt(cauchy, CC_Jaumann)

    ! Computes the contribution of the tangent modulus which is attributed to the Jaumann rate of the 
    ! Kirchhoff stress in ABAQUS. Possesses minor and major symmtries. Independent of the constitutive
    ! model.

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: cauchy      ! Cauchy stress tensor

    real(dp), dimension(6,6), intent(out) :: CC_Jaumann ! Jaumann contribution to the tangent modulus

    CC_Jaumann = zero

    ! main diagonal
    CC_Jaumann(1,1) = two*cauchy(1,1)
    CC_Jaumann(2,2) = two*cauchy(2,2)
    CC_Jaumann(3,3) = two*cauchy(3,3)
    CC_Jaumann(4,4) = (cauchy(1,1)+cauchy(2,2))/two
    CC_Jaumann(5,5) = (cauchy(1,1)+cauchy(3,3))/two
    CC_Jaumann(6,6) = (cauchy(2,2)+cauchy(3,3))/two

    ! off-diagonal
    CC_Jaumann(1,4) = cauchy(1,2)
    CC_Jaumann(1,5) = cauchy(1,3)
    CC_Jaumann(2,4) = cauchy(1,2)
    CC_Jaumann(2,6) = cauchy(2,3)
    CC_Jaumann(3,5) = cauchy(1,3)
    CC_Jaumann(3,6) = cauchy(2,3)
    CC_Jaumann(4,5) = cauchy(2,3)/two
    CC_Jaumann(4,6) = cauchy(1,3)/two
    CC_Jaumann(5,6) = cauchy(1,2)

    ! symmetry
    CC_Jaumann(4,1) = CC_Jaumann(1,4)
    CC_Jaumann(5,1) = CC_Jaumann(1,5)
    CC_Jaumann(4,2) = CC_Jaumann(2,4)
    CC_Jaumann(6,2) = CC_Jaumann(2,6)
    CC_Jaumann(5,3) = CC_Jaumann(3,5)
    CC_Jaumann(6,3) = CC_Jaumann(3,6) 
    CC_Jaumann(5,4) = CC_Jaumann(4,5) 
    CC_Jaumann(6,4) = CC_Jaumann(4,6)
    CC_Jaumann(6,5) = CC_Jaumann(5,6)

    end subroutine jaumann_tangent_voigt
    
    
! ================================================================================================ !  
! ===                          Fictitious non-equilibrium tangent modulus                      === !
! ================================================================================================ !
subroutine fictitious_nonequilibrium_tangent(J_Psi_a, H_Psi_a, T_Psi_a, L, Hbar, Cbar_inv, Q_ra, &
    tau_bar, DTIME, S_ra_bar, Q_ra_old, S_ra_bar_old, J_Tau, CC_ra_bar, CC_ra_neq_bar)

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(1,2,numMaxwell,numTens), intent(in) :: J_Psi_a         ! Jacobian of the free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,2,numMaxwell,numTens), intent(in) :: H_Psi_a       ! Hessian of the free energy w.r.t. the generalized invariants
    real(dp), dimension(1,2,2,2,numMaxwell,numTens), intent(in) :: T_Psi_a     ! Third derivative of the free energy w.r.t. the generalized invariants
    real(dp), dimension(numTens,3,3), intent(in) :: L                          ! generalized strucutral tensor
    real(dp), dimension(numTens,3,3), intent(in) :: Hbar                       ! C_bar_inv*L*C_bar_inv
    real(dp), dimension(3,3), intent(in) :: Cbar_inv                           ! isochoric inverse of right Cauchy-Green deformation tensor
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: Q_ra             ! auxiliary equilibrium stress
    real(dp), dimension(numMaxwell,numTens), intent(in) :: tau_bar             ! average relaxation times during time increment
    real(dp), intent(in) :: DTIME                                              ! time increment
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: S_ra_bar        ! ficticious elastic stress
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: Q_ra_old        ! old auxiliarystresses
    real(dp), dimension(3,3,numMaxwell,numTens), intent(in) :: S_ra_bar_old    ! old ficticious elastic stress
    real(dp), dimension(1,2,numMaxwell,numTens), intent(in) :: J_Tau           ! Jacobians of the relaxation times w.r.t. the invariants
    real(dp), dimension(3,3,3,3,numMaxwell,numTens), intent(in) :: CC_ra_bar   ! fictitious auxiliary non-equilibrium tangent modulus

    real(dp), dimension(3,3,3,3,numMaxwell,numTens), intent(out) ::CC_ra_neq_bar
    
    ! internal variables
    ! ================================================================
    real(dp) :: eps, gamma, xi
    integer :: ii, jj
        
    real(dp), dimension(3,3) :: dtau_dCbar, M, M_T, N, O, delta, QCinv, CinvQ ! , QHbar, HbarQ
        
    real(dp), dimension(3,3,3,3) :: term_1, term_2, term_3,  &
        term_4, term_5, term_6, term_7, term_8, term_9, term_10, term_11 ! , term_8_1
    
    real(dp), dimension(3,3,3,3) :: Cbar_inv_otimes_Cbar_inv_sym, L_odot_L,&
        Hbar_odot_Hbar, Cbar_inv_otimes_Hbar_sym, Hbar_otimes_Cbar_inv_sym, L_odot_Hbar, &
        Hbar_odot_L, Hbar_odot_N, N_odot_Hbar, L_odot_N, N_odot_L, Cbar_inv_otimes_M_sym, &
        M_T_otimes_Cbar_inv_sym, Cbar_inv_otimes_M_T_sym, M_otimes_Cbar_inv_sym, &
        O_otimes_Hbar_sym, Hbar_otimes_O_sym, dQ_ra_dCbar, CC_tmp, CC_1, CC_2, &
        Cinv_otimes_Cinv_sym_QH, HQ_Cinv_otimes_Cinv_sym, KK, KK_QCinv, CinvQ_KK
    
    !call A_otimes_B_sym(Cbar_inv, Cbar_inv, Cbar_inv_otimes_Cbar_inv_sym)
    
    do ii = 1,numTens
        
        ! compute intermediate variables
        call A_odot_B(L(ii,:,:), L(ii,:,:), L_odot_L)
        call A_odot_B(Hbar(ii,:,:), Hbar(ii,:,:), Hbar_odot_Hbar)
        call A_otimes_B_sym(Cbar_inv, Hbar(ii,:,:), Cbar_inv_otimes_Hbar_sym)
        call A_otimes_B_sym(Hbar(ii,:,:), Cbar_inv, Hbar_otimes_Cbar_inv_sym)
        KK = Cbar_inv_otimes_Hbar_sym + Hbar_otimes_Cbar_inv_sym
        call A_odot_B(L(ii,:,:), Hbar(ii,:,:), L_odot_Hbar)
        call A_odot_B(Hbar(ii,:,:), L(ii,:,:), Hbar_odot_L)
        
        do jj = 1,numMaxwell
            
            ! 1.) d/dCbar(CC_ra_bar) : Q_ra 
            ! ================================================================
            call double_contraction_22(Hbar(ii,:,:), Q_ra(:,:,jj,ii), gamma)
            call double_contraction_22(L(ii,:,:), Q_ra(:,:,jj,ii), eps)
            
            M = matmul(matmul(Cbar_inv, Q_ra(:,:,jj,ii)), Hbar(ii,:,:))
            M_T = transpose(M)
            N = M + M_T
            
            O = matmul(matmul(Cbar_inv, Q_ra(:,:,jj,ii)), Cbar_inv)
            
            call A_odot_B(Hbar(ii,:,:), N, Hbar_odot_N)
            call A_odot_B(N, Hbar(ii,:,:), N_odot_Hbar)
            
            call A_odot_B(L(ii,:,:),N, L_odot_N)
            call A_odot_B(N, L(ii,:,:), N_odot_L)
            
            call A_otimes_B_sym(Cbar_inv, M, Cbar_inv_otimes_M_sym)
            call A_otimes_B_sym(M_T, Cbar_inv, M_T_otimes_Cbar_inv_sym)
            
            call A_otimes_B_sym(Cbar_inv, M_T, Cbar_inv_otimes_M_T_sym)
            call A_otimes_B_sym(M, Cbar_inv, M_otimes_Cbar_inv_sym)
            
            call A_otimes_B_sym(O, Hbar(ii,:,:), O_otimes_Hbar_sym)
            call A_otimes_B_sym(Hbar(ii,:,:), O, Hbar_otimes_O_sym)
            
            !QHbar = matmul(Q_ra(:,:,jj,ii), Hbar(ii,:,:))
            !HbarQ = transpose(QHbar)
            !call simple_composition_42(Cbar_inv_otimes_Cbar_inv_sym, QHbar, Cinv_otimes_Cinv_sym_QH)
            !call simple_composition_24(HbarQ, Cbar_inv_otimes_Cbar_inv_sym, HQ_Cinv_otimes_Cinv_sym)
            
            QCinv = matmul(Q_ra(:,:,jj,ii), Cbar_inv)
            CinvQ = transpose(QCinv)
            call simple_composition_42(KK, QCinv, KK_QCinv)
            call simple_composition_24(CinvQ, KK, CinvQ_KK)

            term_1 = T_Psi_a(1,1,1,1,jj,ii)*eps*L_odot_L
            term_2 = T_Psi_a(1,2,2,2,jj,ii)*gamma*Hbar_odot_Hbar
            term_3 = T_Psi_a(1,1,1,2,jj,ii)*eps*L_odot_Hbar
            term_4 = T_Psi_a(1,1,2,2,jj,ii)*gamma*Hbar_odot_L
            term_5 = H_Psi_a(1,2,2,jj,ii)*( gamma*KK + Hbar_odot_N )
            term_6 = H_Psi_a(1,2,2,jj,ii)*N_odot_Hbar
            term_7 = H_Psi_a(1,1,2,jj,ii)*N_odot_L
            
            term_8 = J_Psi_a(1,2,jj,ii)*( Cbar_inv_otimes_M_sym + M_T_otimes_Cbar_inv_sym &
                + Cbar_inv_otimes_M_T_sym + M_otimes_Cbar_inv_sym &
                + O_otimes_Hbar_sym + Hbar_otimes_O_sym )
            
            !term_8_1 = J_Psi_a(1,2,jj,ii)*( Cinv_otimes_Cinv_sym_QH + HQ_Cinv_otimes_Cinv_sym & ! should be identical to term_8
            !    + KK_QCinv + CinvQ_KK)
                        
            term_9  = T_Psi_a(1,1,1,2,jj,ii)*(gamma*L_odot_L + eps*Hbar_odot_L)
            term_10 = T_Psi_a(1,1,2,2,jj,ii)*(gamma*L_odot_Hbar + eps*Hbar_odot_Hbar)
            term_11 = H_Psi_a(1,1,2,jj,ii)*(L_odot_N + eps*KK)

            CC_1 = four*( &
                term_1 - term_2 - term_3 + term_4 - term_5 - term_6 + term_7 - term_8 - term_9 + term_10 + term_11 &
                )


            ! 2.) CC_ra_bar : d/dCbar(Q_ra) 
            ! ================================================================
            ! derivative of tau with respect to Cbar
            dtau_dCbar = J_Tau(1,1,jj,ii)*L(ii,:,:) - J_Tau(1,2,jj,ii)* Hbar(ii,:,:)
            
            ! derivative of Q_ra with respect to Cbar
            xi = -DTIME/(two*tau_bar(jj,ii))
            delta = -xi/tau_bar(jj,ii)*(two*exp(xi)*Q_ra_old(:,:,jj,ii) + S_ra_bar(:,:,jj,ii) - S_ra_bar_old(:,:,jj,ii))
            call A_odot_B(delta, dtau_dCbar, CC_tmp)
            dQ_ra_dCbar = exp(xi)/two*( CC_ra_bar(:,:,:,:,jj,ii) + CC_tmp )
            
            ! contraction of CC_ra_bar and dQ_ra_dCbar
            call composition_fourth_order_AB(CC_ra_bar(:,:,:,:,jj,ii), dQ_ra_dCbar, CC_2)
            
            
            ! 3.) CC_ra_neq_bar = d/dCbar(CC_ra_bar) : Q_ra + CC_ra_bar : d/dCbar(Q_ra) 
            ! ================================================================
            CC_ra_neq_bar(:,:,:,:,jj,ii) = CC_1 + CC_2

        end do
    end do
    
end subroutine fictitious_nonequilibrium_tangent
    
    
! ================================================================================================ !  
! ===                                        Determinant                                       === !
! ================================================================================================ !      
subroutine determinant(F, det)

    ! calculate the determinat of a 3x3 matrix F using the rule of Sarrus

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: F
    real(dp), intent(out) :: det
 
    det =   F(1,1)*F(2,2)*F(3,3) &
          + F(1,2)*F(2,3)*F(3,1) &
          + F(1,3)*F(3,2)*F(2,1) &
          - F(1,3)*F(3,1)*F(2,2) &
          - F(2,3)*F(3,2)*F(1,1) &
          - F(1,2)*F(2,1)*F(3,3)
      
end subroutine determinant


! ================================================================================================ !  
! ===                                           Cofactor                                       === !
! ================================================================================================ !  
subroutine cofactor(A, cofA)

    ! calculate the cofactor matrix of 3x3 matrix A

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A
    real(dp), dimension(3,3), intent(out) :: cofA

    cofA(1,1) = +(A(2,2)*A(3,3)-A(2,3)*A(3,2))
    cofA(1,2) = -(A(2,1)*A(3,3)-A(2,3)*A(3,1))
    cofA(1,3) = +(A(2,1)*A(3,2)-A(2,2)*A(3,1))
    cofA(2,1) = -(A(1,2)*A(3,3)-A(1,3)*A(3,2))
    cofA(2,2) = +(A(1,1)*A(3,3)-A(1,3)*A(3,1))
    cofA(2,3) = -(A(1,1)*A(3,2)-A(1,2)*A(3,1))
    cofA(3,1) = +(A(1,2)*A(2,3)-A(1,3)*A(2,2))
    cofA(3,2) = -(A(1,1)*A(2,3)-A(1,3)*A(2,1))
    cofA(3,3) = +(A(1,1)*A(2,2)-A(1,2)*A(2,1))

end subroutine cofactor
    
      
! ================================================================================================ !  
! ===                                            Inverse                                       === !
! ================================================================================================ !  
subroutine inverse(A, invA)

    ! computes the inverse of a 3x3 matrix A

    use precision
    implicit none

    ! formal varaiables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A
    real(dp), dimension(3,3), intent(out) :: invA    

    ! internal variables
    ! ================================================================
    real(dp), dimension(3,3) :: cofA
    real(dp) :: detA

    call determinant(A, detA)
    call cofactor(A, cofA)

    invA = transpose(cofA)/detA

end subroutine inverse
    

! ================================================================================================ !  
! ===                                 Fourth-order identity tensor                             === !
! ================================================================================================ !  
subroutine fourth_order_identity(eye_2, eye_4)

    ! Computes the fourth-order identity tensor eye_4 = eye_2 \otimes eye_2
    ! Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l)

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: eye_2       ! second-order identity tensor
    real(dp), dimension(3,3,3,3), intent(out) :: eye_4  ! fourth-order identity tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    eye_4(ii,jj,kk,ll) = eye_2(ii,kk)*eye_2(ll,jj)
                end do
            end do
        end do
    end do

end subroutine fourth_order_identity
  

! ================================================================================================ !  
! ===                          Supersymmetric fourth-order identity tensor                     === !
! ================================================================================================ !      
subroutine sym_fourth_order_eye(eye_2, eye_4_sym)

    ! Computes the fourth-order identity tensor eye_4_sym = 1/2*( eye_4 + eye_4^t)
    ! Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l)

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: eye_2           ! second-order identity tensor
    real(dp), dimension(3,3,3,3), intent(out) :: eye_4_sym  ! fourth-order identity tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    eye_4_sym(ii,jj,kk,ll) = ( eye_2(ii,kk)*eye_2(ll,jj) + eye_2(ii,ll)*eye_2(kk,jj) ) / 2.d0
                end do
            end do
        end do
    end do

    end subroutine sym_fourth_order_eye
    
    
! ================================================================================================ !  
! ===                                    Fourth-order transpose                                === !
! ================================================================================================ !  
subroutine transpose_fourth_order(A, transpose_A)

    ! Compute the transpose of a fourth-order tensor A whose components are given w.r.t. the basis
    ! (e_i \otimes e_j) \odot (e_k \otimes e_l). transA_{ijkl} = A_{klij}

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,3,3), intent(in) :: A            ! fourth-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: transpose_A ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    transpose_A(ii,jj,kk,ll) = A(kk,ll,ii,jj)
                end do
            end do
        end do
    end do

end subroutine transpose_fourth_order
    
    
! ================================================================================================ !  
! ===                                 Deviatotric projection tensor                            === !
! ================================================================================================ !  
subroutine deviatoric_projection_tensor(eye, C, Cinv, P_Dev)
    
    ! Computes the Lagrangian deviatoric projection tensor P_Dev = eye_4_sym - 1/3*C^{-1} \odot C
    ! Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l)

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: eye ! second-order identity tensor
    real(dp), dimension(3,3), intent(in) :: C, Cinv ! right Cauchy-Green deformation tensor and its inverse
    real(dp), dimension(3,3,3,3), intent(out) :: P_Dev ! Lagrangian deviatoric projection tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    P_Dev(ii,jj,kk,ll) =  ( eye(ii,kk)*eye(ll,jj) + eye(ii,ll)*eye(kk,jj) ) / 2.d0 - Cinv(ii,jj)*C(kk,ll)/3.d0
                end do
            end do
        end do
    end do

end subroutine deviatoric_projection_tensor

    
! ================================================================================================ !  
! ===                                           A \odot B                                      === !
! ================================================================================================ !  
subroutine A_odot_B(A, B, C)

    ! Computes the tensor product C = A \odot B of two second-order tensors A and B yielding a fourth-order
    ! tensor C. Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l): C_{ijkl} = A_{ij} B_{kl}.

    use precision
    implicit none

    ! formal variables 
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A, B    ! second-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: C  ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll


    do ll = 1,3
        do kk = 1,3
            C(:,:,kk,ll) = A*B(kk,ll)
        end do
    end do

end subroutine A_odot_B
    

! ================================================================================================ !  
! ===                                         (A \otimes B)                                    === !
! ================================================================================================ !     
subroutine A_otimes_B(A, B, C)

    ! Computes the tensor product C = A \otimes B of two second-order tensors A and B yielding a fourth-order tensor C.
    ! Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l):  C_{ijkl} = A_{ik}B_{lj}.

    use precision
    implicit none

    ! formal variables 
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A, B    ! second-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: C  ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    C(ii,jj,kk,ll) = A(ii,kk)*B(ll,jj)
                end do
            end do
        end do
    end do

end subroutine A_otimes_B

    
! ================================================================================================ !  
! ===                                        (A \otimes B)^s                                   === !
! ================================================================================================ !     
subroutine A_otimes_B_sym(A, B, C)

    ! Computes the tensor product with subsequent symmetrization C = (A \otimes B)^s of two second-order
    ! tensors A and B yielding a fourth-order tensor C.
    ! Compoents are given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l):  C_{ijkl} = (A_{ik}B_{lj} + A_{il} B_{kj})/2.

    use precision
    implicit none

    ! formal variables 
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A, B    ! second-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: C  ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    C(ii,jj,kk,ll) = (A(ii,kk)*B(ll,jj) + A(ii,ll)*B(kk,jj))/2.d0
                end do
            end do
        end do
    end do

end subroutine A_otimes_B_sym
   
    
! ================================================================================================ !  
! ===                                     Tensor scalar product                                === !
! ================================================================================================ !  
subroutine double_contraction_22(A, B, c)

    ! Computes the double contraction (tensor scalar product) of two second-order tensors A and B giving a scalar c

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: A, B ! second-order tensor
    real(dp), intent(out) :: c                   ! scalar

    !internal variables 
    ! ================================================================
    integer :: ii, jj

    c = 0.d0
    do jj = 1,3
        do ii = 1,3
            c = c + A(ii,jj)*B(ii,jj)
        end do
    end do

end subroutine double_contraction_22
    
    
! ================================================================================================ !  
! ===                     Double contraction of fourth- and second-oder tensor                 === !
! ================================================================================================ !  
subroutine double_contraction_42(A, B, C)

    ! Computes the double contraction of a fourth-order tensors A and a second-order tensor B giving
    ! second-order tensor C

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,3,3), intent(in) :: A  ! fourth-order tensor
    real(dp), dimension(3,3), intent(in) :: B      ! fourth-order tensor
    real(dp), dimension(3,3), intent(out) :: C     ! fourth-order tensor

    ! internal variables 
    ! ================================================================
    integer :: ii, jj, kk, ll
    real(dp) :: tmp

    do jj = 1,3
        do ii = 1,3
            tmp = 0.d0
            do ll = 1,3
                do kk = 1,3
                    tmp = tmp + A(ii,jj,kk,ll)*B(kk,ll)
                end do
            end do
            C(ii,jj) = tmp
        end do
    end do

end subroutine double_contraction_42
    

! ================================================================================================ !  
! ===                                      Simple composition                                  === !
! ================================================================================================ !  
subroutine simple_composition_42(A, B, C)

    ! Computes the simple composition C = A . B of a fourth-order tensor A, given w.r.t. the basis
    ! (e_i \otimes e_j) \odot (e_k \otimes e_l), and a second-order tensor B, given w.r.t. the basis 
    ! (e_m \otimes e_n).

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in)  :: A(3,3,3,3)   ! A(i,n,k,l)
    real(dp), intent(in)  :: B(3,3)       ! B(n,j)
    real(dp), intent(out) :: C(3,3,3,3)   ! C(i,j,k,l)

    ! internal variables
    ! ================================================================
    integer :: ll, kk, jj, nn, ii
    real(dp) :: s

    C = 0.0d0
    do ll = 1, 3
        do kk = 1, 3
            do jj = 1, 3
                do nn = 1, 3
                    s = B(nn, jj)                 ! reuse this scalar across i
                    if (s /= 0.0d0) then
                        do ii = 1, 3                 ! unit-stride access
                            C(ii, jj, kk, ll) = C(ii, jj, kk, ll) + A(ii, nn, kk, ll) * s
                        end do
                    end if
                end do
            end do
        end do
    end do
  
end subroutine simple_composition_42
    
    
! ================================================================================================ !  
! ===                                      Simple composition                                  === !
! ================================================================================================ !  
subroutine simple_composition_24(B, A, C)

    ! Computes the simple composition C = B . A of a fourth-order tensor A, given w.r.t. the basis
    ! (e_i \otimes e_j) \odot (e_k \otimes e_l), and a second-order tensor B, given w.r.t. the basis 
    ! (e_m \otimes e_n).

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), intent(in)  :: A(3,3,3,3)   ! A(i,n,k,l)
    real(dp), intent(in)  :: B(3,3)       ! B(n,j)
    real(dp), intent(out) :: C(3,3,3,3)   ! C(i,j,k,l)

    ! internal variables
    ! ================================================================
    integer :: ll, kk, jj, nn, ii
    real(dp) :: s

    C = 0.0d0
    do ll = 1, 3
        do kk = 1, 3
            do jj = 1, 3
                do nn = 1, 3
                    s = A(nn, jj, kk, ll)                 ! reuse this scalar across i
                    if (s /= 0.0d0) then
                        do ii = 1, 3                     ! unit-stride in first dim
                            C(ii, jj, kk, ll) = C(ii, jj, kk, ll) + B(ii, nn) * s
                        end do
                    end if
                end do
            end do
        end do
    end do
    
end subroutine simple_composition_24
    
    
! ================================================================================================ !  
! ===                                   Fourth-order composition                               === !
! ================================================================================================ !  
subroutine composition_fourth_order_ABC(A, B, C, D)

    ! Computes the composition D = A : B : C of three fourth-order tensor A, B and C whose components are
    ! given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l).
    ! Components: D_{ijkl} = A_{ijmn}*B_{mnrs}*C_{rskl}.
    ! This composition is is frequently used for push-forward, pull-back or projection operations of fourth order tensors

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,3,3), intent(in) :: A, B, C ! fourth-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: D      ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll, mm, nn, oo, pp
    real(dp), dimension(3,3) :: tmpm
    real(dp), dimension(3,3,3,3) :: A1
    
    D = 0d0
    do oo=1,3
        do pp=1,3
            tmpm = 0d0
            do mm=1,3
                do nn=1,3
                    tmpm = tmpm + A(:, :, mm, nn)*B(mm, nn, oo, pp)
                end do
            end do
            A1(:, :, oo, pp) = tmpm
        end do
    end do

    do oo=1,3
        do pp=1,3
            tmpm = 0d0
            do mm=1,3
                do nn=1,3
                    tmpm = tmpm + A1(:, :, mm, nn)*C(mm, nn, oo, pp)
                end do
            end do
            D(:, :, oo, pp) = tmpm
        end do
    end do

end subroutine composition_fourth_order_ABC


! ================================================================================================ !  
! ===                                   Fourth-order composition                               === !
! ================================================================================================ !  
subroutine composition_fourth_order_AB(A, B, C)

    ! Computes the composition C = A : B of two fourth-order tensor A and B whose components are
    ! given w.r.t. the basis (e_i \otimes e_j) \odot (e_k \otimes e_l).
    ! Components: C_{ijkl} = A_{ijmn}*B_{mnkl}.

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,3,3), intent(in) :: A, B  ! fourth-order tensor
    real(dp), dimension(3,3,3,3), intent(out) :: C    ! fourth-order tensor

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll, mm, nn
    real(dp) tmp

    do ll = 1,3
        do kk = 1,3
            do jj = 1,3
                do ii = 1,3
                    tmp = 0.0d0
                    do nn = 1,3
                        do mm = 1,3
                            tmp = tmp + A(ii,jj,mm,nn)*B(mm,nn,kk,ll)
                        end do
                    end do
                    C(ii,jj,kk,ll) = tmp
                end do
            end do
        end do
    end do

end subroutine composition_fourth_order_AB    

 
! ================================================================================================ !  
! ===                Convert Cauchy stress to Voigt notation suitable for ABAQUS               === !
! ================================================================================================ !   
subroutine voigt_notation_stress(cauchy, NTENS, indices, STRESS)

    ! cauchy(3,3) --> STRESS(NTENS)

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3), intent(in) :: cauchy     ! Cauchy stress tensor
    integer, intent(in) :: NTENS                               ! Number of stress components
    integer, dimension(2,NTENS), intent(in) :: indices         ! Voigt indices

    real(dp), dimension(NTENS), intent(out) :: STRESS  ! Cauchy stress tensor in Voigt notation

    ! internal variables
    ! ================================================================
    integer :: ii, jj, nn

    ! convert the stress
    ! ================================================================
    do nn = 1,NTENS
	    ii=indices(1,nn)
	    jj=indices(2,nn)
	    STRESS(nn) = cauchy(ii,jj)
    end do

end subroutine voigt_notation_stress

    
    
! ================================================================================================ !  
! ===              Convert tangent modulus to Voigt notation suitable for ABAQUS               === !
! ================================================================================================ !   
subroutine voigt_notation_tangent(CC_tot, NTENS, indices, DDSDDE)

    ! CC_tot(3,3,3,3) --> DDSDDE(NTENS,NTENS)

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(3,3,3,3), intent(in) :: CC_tot       ! tangent modulus
    integer, intent(in) :: NTENS                                     ! Number of stress components
    integer, dimension(2,NTENS), intent(in) :: indices               ! Voigt indices

    real(dp), dimension(NTENS,NTENS), intent(out) :: DDSDDE  ! Tangent modulus in Voigt notation

    ! internal variables
    ! ================================================================
    integer :: ii, jj, kk, ll, mm, nn


    ! convert the tangent
    ! ================================================================
    do mm = 1,NTENS
	    ii = indices(1,mm)
	    jj = indices(2,mm)   
	    do nn = 1,NTENS
		    kk = indices(1,nn)
		    ll = indices(2,nn)
		    DDSDDE(mm,nn) = CC_tot(ii,jj,kk,ll)
	    end do
    end do

end subroutine voigt_notation_tangent


    
! ================================================================================================ !  
! ===                                  Voigt to tensor notation                                === !
! ================================================================================================ !   
subroutine voigt2tens(vector, tensor)

    ! Converts a symmetric second-order from Voigt notation (1,6) to standard tensor notation (3,3)

    use precision
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(6), intent(in) :: vector
    real(dp), dimension(3,3), intent(out) :: tensor

    ! convert vector to symmetric tensor
    ! ================================================================

    ! Diagonal components
    tensor(1,1) = vector(1)
    tensor(2,2) = vector(2)
    tensor(3,3) = vector(3)

    ! Off-diagonal compontens
    tensor(1,2) = vector(4)
    tensor(3,1) = vector(5)
    tensor(2,3) = vector(6)

    ! Complete the tensor exploiting its symmetry
    tensor(2,1) = tensor(1,2)
    tensor(1,3) = tensor(3,1)
    tensor(3,2) = tensor(2,3)

end subroutine voigt2tens



! ================================================================================================ !  
! ===                             Voigt to tangent in tensor notation                          === !
! ================================================================================================ !
subroutine voigt2tensor_tangent(V,T)

    ! converts the tangent modulus in Voigt notation to tensor notation

    use precision
    implicit none

    real(dp), dimension(6,6), intent(in) :: V   ! tangent in Voigt notation
    real(dp), dimension(3,3,3,3), intent(out) :: T  ! tangent in tensor notation

    ! ===
    T(1,1,1,1) = V(1,1)
    T(2,2,2,2) = V(2,2)
    T(3,3,3,3) = V(3,3)
    T(1,1,2,2) = V(1,2)
    T(1,1,3,3) = V(1,3)
    T(2,2,3,3) = V(2,3)
    T(2,2,1,1) = V(2,1)
    T(3,3,2,2) = V(3,2)
    T(3,3,1,1) = V(3,1)

    ! ===
    T(1,1,1,2) = V(1,4); T(1,1,2,1) = V(1,4) 
    T(1,1,1,3) = V(1,5); T(1,1,3,1) = V(1,5) 
    T(1,1,2,3) = V(1,6); T(1,1,3,2) = V(1,6) 

    T(2,2,1,2) = V(2,4); T(2,2,2,1) = V(2,4) 
    T(2,2,1,3) = V(2,5); T(2,2,3,1) = V(2,5)
    T(2,2,2,3) = V(2,6); T(2,2,3,2) = V(2,6) 

    T(3,3,1,2) = V(3,4); T(3,3,2,1) = V(3,4) 
    T(3,3,1,3) = V(3,5); T(3,3,3,1) = V(3,5)
    T(3,3,2,3) = V(3,6); T(3,3,3,2) = V(3,6)

    ! ===
    T(1,2,1,1) = V(4,1); T(2,1,1,1) = V(4,1) 
    T(1,2,2,2) = V(4,2); T(2,1,2,2) = V(4,2)
    T(1,2,3,3) = V(4,3); T(2,1,3,3) = V(4,3)

    T(1,3,1,1) = V(5,1); T(3,1,1,1) = V(5,1) 
    T(1,3,2,2) = V(5,2); T(3,1,2,2) = V(5,2)
    T(1,3,3,3) = V(5,3); T(3,1,3,3) = V(5,3)

    T(2,3,1,1) = V(6,1); T(3,2,1,1) = V(6,1) 
    T(2,3,2,2) = V(6,2); T(3,2,2,2) = V(6,2)
    T(2,3,3,3) = V(6,3); T(3,2,3,3) = V(6,3)

    !===
    T(1,2,1,2) = V(4,4); T(2,1,1,2) = V(4,4); T(1,2,2,1) = V(4,4); T(2,1,2,1) = V(4,4)
    T(1,2,1,3) = V(4,5); T(2,1,1,3) = V(4,5); T(1,2,3,1) = V(4,5); T(2,1,3,1) = V(4,5)
    T(1,2,2,3) = V(4,6); T(2,1,2,3) = V(4,6); T(1,2,3,2) = V(4,6); T(2,1,3,2) = V(4,6)

    T(1,3,1,2) = V(5,4); T(3,1,1,2) = V(5,4); T(1,3,2,1) = V(5,4); T(3,1,2,1) = V(5,4) 
    T(1,3,1,3) = V(5,5); T(3,1,1,3) = V(5,5); T(1,3,3,1) = V(5,5); T(3,1,3,1) = V(5,5) 
    T(1,3,2,3) = V(5,6); T(3,1,2,3) = V(5,6); T(1,3,3,2) = V(5,6); T(3,1,3,2) = V(5,6) 

    T(2,3,1,2) = V(6,4); T(3,2,1,2) = V(6,4); T(2,3,2,1) = V(6,4); T(3,2,2,1) = V(6,4) 
    T(2,3,1,3) = V(6,5); T(3,2,1,3) = V(6,5); T(2,3,3,1) = V(6,5); T(3,2,3,1) = V(6,5) 
    T(2,3,2,3) = V(6,6); T(3,2,2,3) = V(6,6); T(2,3,3,2) = V(6,6); T(3,2,3,2) = V(6,6) 

end subroutine voigt2tensor_tangent



! ================================================================================================ !  
subroutine compute_Fdot(F0, F1, DTIME, Fdot)

    ! compute the material time derivative of the deformation gradient using forward differences

    use precision
    implicit none

    real(dp), dimension(3,3), intent(in) :: F0, F1
    real(dp), intent(in) :: DTIME
    
    real(dp), dimension(3,3), intent(out) :: Fdot

    Fdot = (F1-F0)/DTIME

    end subroutine compute_Fdot

! ================================================================================================ !  

subroutine compute_Cdot(F, Fdot, Cdot)

    ! compute the material time derivative of the right Cauchy Green deformation tensor using forward differences

    use precision
    implicit none

    real(dp), dimension(3,3), intent(in) :: F, Fdot
    
    real(dp), dimension(3,3), intent(out) :: Cdot

    Cdot = matmul(transpose(F), Fdot) + matmul(transpose(Fdot), F) 

end subroutine compute_Cdot

! ================================================================================================ !  
! ===                              Closed-form polar decomposition                             === !
! ================================================================================================ !  
    
subroutine polardecomp_closed_form(F,R,U)

    ! This subroutine computes the right polar decomposition of a matrix
    ! [F] into an orthogonal rotation matrix [R] and a symmetric matrix
    ! [U]. Based on the algorithm by Simo et al. (2000)
    
    use precision
    implicit none
    ! Input/Output
    real(dp) F(3,3), R(3,3), U(3,3)

    integer ii,jj
    real(dp) IC, IIC, IIIC ! Invariants of C 
    real(dp) IU, IIU, IIIU ! Invariants of U
    real(dp) l1,l2,l3 ! eigenvalues
    real(dp) p,q,m,n,t,D ! constants
    real(dp) I(3,3) ! Identity
    real(dp) C(3,3),CC(3,3) ! right Cauchy-Green, squared
    real(dp) Ui(3,3) ! U inverse
    ! Parameters
    real(dp) zero, one, two, three, pi, tol
    parameter ( zero=0d0, one=1d0, two=2d0, three=3d0, &
                pi=3.1415926535897932d0, tol=1d-8 )

    do ii=1,3
        do jj=1,3
            if (ii.eq.jj) then
                I(ii,jj) = one
            else
                I(ii,jj) = zero
            end if
        end do
    end do

    ! Right Cauchy-Green
    C(:,:)  = matmul(transpose(F),F)
    CC(:,:) = matmul(C,C)

    ! Invariants
    IC = C(1,1) + C(2,2) + C(3,3)
    IIC = 5d-1*(IC**2  - (CC(1,1) + CC(2,2) + CC(3,3)))
    IIIC = C(1,1) * (C(2,2)*C(3,3) - C(2,3)*C(3,2)) &
         + C(1,2) * (C(2,3)*C(3,1) - C(2,1)*C(3,3)) &
         + C(1,3) * (C(2,1)*C(3,2) - C(2,2)*C(3,1))

    ! Eigenvalues of sqrt(C)
    p = IIC - (IC**two)/three
    q = -(two/27d0)*IC**3+IC*IIC/three-IIIC
    if( abs(p)<tol )then
    l1 = sqrt( abs(-abs(q)**(one/three) + IC/three) )
    l2 = l1
    l3 = l2
    else
    m = two*sqrt(abs(p)/three)
    n = three*q/(m*p)
    if(abs(n)>one) n = sign(one, n)
    t = atan2(sqrt(one-n**2),n)/three
    l1 = sqrt( abs(m*cos(t) + IC/three) )
    l2 = sqrt( abs(m*cos(t+two/three*pi) + IC/three) )
    l3 = sqrt( abs(m*cos(t+4d0/three*pi) + IC/three) )
    endif

    ! Invariants of stretch
    IU = l1 + l2 + l3
    IIU = l1*l2 + l1*l3 + l2*l3
    IIIU = l1*l2*l3
    D = IU*IIU-IIIU

    ! Stretch and inverse
    U(:,:)  = (-CC(:,:) + (IU**2-IIU)*C(:,:) + IU*IIIU*I(:,:))/D
    Ui(:,:) = (C(:,:) - IU*U(:,:) + IIU*I(:,:))/IIIU
    ! Rotation
    R(:,:) = matmul(F,Ui)

    return
    
end subroutine polardecomp_closed_form

! ================================================================================================ !
! ===                        Subroutine Generalized Structural Tensors                         === !
! ================================================================================================ !

subroutine vCANN_structural(extra_in, material_dirs, struc_weights) 

use precision 
use parameters 
!use activation_functions 

implicit none 

! formal variables 
! ================================================================ 
real(dp), dimension(1,1), intent(in) :: extra_in                   ! input feature to structure learning 
real(dp), dimension(numDir,3), intent(out) :: material_dirs        ! preferred material directions 
real(dp), dimension(numTens,numDir+1), intent(out) ::  struc_weights ! scalar weights of the preferred material directions 

! Nothing to compute here since the material is isotropic!

end subroutine vCANN_structural



! ================================================================================================ !
! ===                                   Subroutine UEXTERNALDB                                 === !
! ================================================================================================ !

subroutine UEXTERNALDB(LOP,LRESTART,TIME,DTIME,KSTEP,KINC) 

! Loads the vCANN kernels and biases and stores them accessible for each subroutine in the "parameters" module 

  use parameters 
  use vCANNs 
  implicit none 

  ! formal variables 
  ! ================================================================ 
  integer :: LOP, LRESTART, KSTEP, KINC 
  double precision :: TIME, DTIME 
  dimension :: TIME(2) 

!  do while(myVar /= 999) 
!      myVar = 1 
!  end do 

  if (LOP == 0) then 
    call init_weights_eq() 
    call init_weights_neq() 
    call init_offset() 
  end if 

end subroutine UEXTERNALDB

! ================================================================================================ !
! ===                                     Subroutine SDVINI                                    === !
! ================================================================================================ !

subroutine SDVINI(STATEV,COORDS,NSTATV,NCRDS,NOEL,NPT,LAYER,KSPT) 

! Calculates the generalized structural tensors at the beginning of an analysis for each elementuse parameters 
implicit none 
! formal variables 
! ================================================================ 
integer :: NSTATV, NCRDS, NOEL, NPT, LAYER, KSPT 
double precision :: STATEV, COORDS 
dimension STATEV(NSTATV), COORDS(NCRDS) 
END subroutine SDVINI 

