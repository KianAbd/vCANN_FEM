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
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_neq_bar  ! fictirious non-equilibrium 2. PK stresses
real(dp), dimension(3,3,numMaxwell,numTens) :: S_ra_neq      ! deviatoric non-equilibrium 2. PK stresses

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
end if

! === Get state variables from previous time step
! ================================================================
call get_statevs(STATEV, S_ra_bar_old, Q_ra_old, tau_old, numMaxwell, numTens, NSTATV)

! === STRUCTURAL TENSORS
! ================================================================
call structural_tensors(L) ! compute the generalized structural tensors

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

! 4.1   Add all contributions
! ================================================================

! === Actual stresses
S_eq = sum(S_eq_r, dim=3)                  ! equilibirum stress
S_neq  = sum(sum(S_ra_neq, dim=4),dim=3)   ! non-equilibrium stress
S_iso = S_eq + S_neq                       ! total isochoric stress

! === Fictitious stresses
S_eq_bar = sum(S_eq_r_bar, dim=3)                  ! fictitious equilibrium stress
S_neq_bar  = sum(sum(S_ra_neq_bar, dim=4),dim=3)   ! fictitious non-equilibrium stress
S_bar = S_eq_bar + S_neq_bar                       ! total fictitious stress

! 4.2   Apply incompressibility constraint for debugging in Visual Studio and for shell and plane stress elements in ABAQUS
! ================================================================
if (NTENS .EQ. 3) then
    S = S_iso - S_iso(3,3)/Cinv(3,3)*Cinv
else
    S = S_iso
end if

! 4.3   Push-forward to Cauchy stress tensor
! ================================================================
call cauchy_stress(S, F, det, cauchy)

! 4.4   Kirchhoff stress tensor
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

    ! 1.   Equilibrium and auxiliary non-equilibrium contribution
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
!STATEV(40) = DFGRD1(1,1)
!STATEV(41) = DFGRD1(1,2)
!STATEV(42) = DFGRD1(1,3)
!STATEV(43) = DFGRD1(2,1)
!STATEV(44) = DFGRD1(2,2)
!STATEV(45) = DFGRD1(2,3)
!STATEV(46) = DFGRD1(3,1)
!STATEV(47) = DFGRD1(3,2)
!STATEV(48) = DFGRD1(3,3)

end subroutine UMAT

! ================================================================
! ===                        END UMAT                          ===
! ================================================================

    
    
! ================================================================================================ !  
! ===                                      Strucutral tensors                                  === !
! ================================================================================================ !      
subroutine structural_tensors(L)

    use precision
    use parameters
    implicit none

    ! formal variables
    ! ================================================================
    real(dp), dimension(numTens,3,3), intent(out) :: L    ! generalized structural tensors

    ! internal varibales
    ! ================================================================
    real(dp), dimension(numDir,3) :: material_dirs        ! preferred material directions
    real(dp), dimension(3) :: dir                         ! preferred material directions
    real(dp), dimension(numTens,numDir+1) :: dir_weights  ! scalar weighting factors of the generalized structural tensors
    real(dp), dimension(3,3) :: L_0                       ! isotropic structural tensor
    real(dp), dimension(3,3,numDir) :: L_rj               ! structural tensors
    integer :: jj, ii

    L_0 = eye/three ! isotropic contribution

    if (numDir == 0) then
        do jj = 1,numTens
            L(jj,:,:) = L_0
        end do
    else
        ! preferred material directions and scalar weights (should be calculated in the beginning)
        ! ================================================================
        call vCANN_structural(extra_struc, material_dirs, dir_weights)
        
        ! structural tenors
        ! ================================================================
        do ii = 1,numDir
            dir = material_dirs(ii,:)
            L_rj(:,:,ii) = spread(dir,dim=2,ncopies=3) * spread(dir,dim=1,ncopies=3)
        end do

        ! generalized structural tensors
        ! ================================================================
        L = zero
        do jj = 1,numTens
            L(jj,:,:) = L_0*dir_weights(jj,1)
            do ii = 1,numDir
                L(jj,:,:) = L(jj,:,:) + L_rj(:,:,ii)*dir_weights(jj,ii+1)
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
    ! ================================================================
    real(dp) :: eps, gamma, xi, tau_eff, fac
    real(dp), parameter :: eps_tau = 1.0d-14    ! --- Safety floor for tau to avoid division by (near) zero --------------------
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
            tau_eff = max(tau_bar(jj,ii), eps_tau)
            xi = -DTIME/(two*tau_eff)
            fac = -xi/tau_eff
            delta = fac * (two*exp(xi)*Q_ra_old(:,:,jj,ii) + S_ra_bar(:,:,jj,ii) - S_ra_bar_old(:,:,jj,ii))
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