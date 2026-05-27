module forward_pass
  use act_derivs
  implicit none
  private
  public :: forward_with_derivs

contains

  !------------------------------------------------------------------
  ! Forward pass: computes x^l, z^l, and φ′, φ″, φ‴ at each layer
  !------------------------------------------------------------------
  subroutine forward_with_derivs(weights, biases, acts, x, x_list, z_list, d1_list, d2_list, d3_list, L)
    implicit none
    ! Inputs
    real(8), intent(in)  :: weights(:,:,:)
    real(8), intent(in)  :: biases(:,:)   
    integer, intent(in)  :: acts(:)
    real(8), intent(in)  :: x(:)
    integer, intent(in)  :: L

    ! Outputs
    real(8), allocatable, intent(out) :: x_list(:,:), z_list(:,:), d1_list(:,:), d2_list(:,:), d3_list(:,:)

    ! Locals
    integer :: ll, n_in, n_out, max_out
    real(8), allocatable :: z(:), d1(:), d2(:), d3(:), x_next(:)

    ! Find maximum layer width for allocation convenience
    max_out = size(weights,1)

    ! Allocate lists
    allocate(x_list(max_out,L+1))
    allocate(z_list(max_out,L))
    allocate(d1_list(max_out,L))
    allocate(d2_list(max_out,L))
    allocate(d3_list(max_out,L))

    ! Store input as x^0
    x_list(:,1) = 0.0d0
    x_list(1:size(x),1) = x

    ! Loop over layers
    do ll = 1,L
       n_in  = size(weights,2)
       n_out = size(weights,1)

       allocate(z(n_out), d1(n_out), d2(n_out), d3(n_out), x_next(n_out))

       ! Linear transform: z^l = W^l x^{l-1} + b^l
       z = matmul(weights(:,:,ll), x_list(1:n_in,ll)) + biases(1:n_out,ll)

       ! Store preactivation
       z_list(:,ll) = 0.0d0
       z_list(1:n_out,ll) = z

       ! Activation-specific derivatives
       select case(acts(ll))
       case(1)   ! tanh
          call tanh_derivs(z, x_next, d1, d2, d3)
       case(2)   ! sigmoid
          call sigmoid_derivs(z, x_next, d1, d2, d3)
       case(3)   ! softplus
          call softplus_derivs(z, x_next, d1, d2, d3)
       case default
          stop "Unsupported activation code"
       end select

       ! Store results
       d1_list(:,ll) = 0.0d0; d2_list(:,ll) = 0.0d0; d3_list(:,ll) = 0.0d0
       d1_list(1:n_out,ll) = d1
       d2_list(1:n_out,ll) = d2
       d3_list(1:n_out,ll) = d3

       x_list(:,ll+1) = 0.0d0
       x_list(1:n_out,ll+1) = x_next

       ! Deallocate temporaries
       deallocate(z, d1, d2, d3, x_next)
    end do

  end subroutine forward_with_derivs

end module forward_pass

   
    

module act_derivs
  implicit none
  private
  public :: tanh_derivs, sigmoid_derivs, softplus_derivs

contains

  !-----------------------------------------------------------
  ! tanh: φ(z) = t, d1 = 1 - t^2
  !       d2 = -2 t (1 - t^2)
  !       d3 = -2 (1 - t^2) (1 - 3 t^2)
  !-----------------------------------------------------------
  subroutine tanh_derivs(z, x, d1, d2, d3)
    implicit none
    real(8), intent(in)  :: z(:)
    real(8), intent(out) :: x(:), d1(:), d2(:), d3(:)
    real(8), allocatable :: one_minus_t2(:)
    integer :: n

    n = size(z)
    allocate(one_minus_t2(n))

    x  = tanh(z)
    one_minus_t2 = 1.0d0 - x*x

    d1 = one_minus_t2
    d2 = -2.0d0 * x * one_minus_t2
    d3 = -2.0d0 * one_minus_t2 * (1.0d0 - 3.0d0*x*x)

    deallocate(one_minus_t2)
  end subroutine tanh_derivs


  !-----------------------------------------------------------
  ! Sigmoid φ(z) = s, stable form.
  !   d1 = s(1-s)
  !   d2 = s(1-s)(1-2s)
  !   d3 = s(1-s)(1 - 6s + 6s^2)
  !-----------------------------------------------------------
  subroutine sigmoid_derivs(z, x, d1, d2, d3)
    implicit none
    real(8), intent(in)  :: z(:)
    real(8), intent(out) :: x(:), d1(:), d2(:), d3(:)
    logical, allocatable :: mask(:)
    integer :: n

    n = size(z)
    allocate(mask(n))

    ! Stable computation of sigmoid:
    mask = (z >= 0.0d0)
    x    = 0.0d0
    where (mask)
      x = 1.0d0 / (1.0d0 + exp(-z))
    elsewhere
      x = exp(z) / (1.0d0 + exp(z))
    end where

    d1 = x * (1.0d0 - x)
    d2 = d1 * (1.0d0 - 2.0d0*x)
    d3 = d1 * (1.0d0 - 6.0d0*x + 6.0d0*x*x)

    deallocate(mask)
  end subroutine sigmoid_derivs


  !-----------------------------------------------------------
  ! Softplus φ(z) = log(1+exp(z)), φ′ = s = sigmoid(z)
  !   d1 = s
  !   d2 = s(1-s)
  !   d3 = d2(1-2s)
  !-----------------------------------------------------------
  subroutine softplus_derivs(z, x, d1, d2, d3)
    implicit none
    real(8), intent(in)  :: z(:)
    real(8), intent(out) :: x(:), d1(:), d2(:), d3(:)
    logical, allocatable :: mask(:)
    integer :: n

    n = size(z)
    allocate(mask(n))

    ! Compute sigmoid s, reuse as d1
    mask = (z >= 0.0d0)
    d1   = 0.0d0
    where (mask)
      d1 = 1.0d0 / (1.0d0 + exp(-z))
    elsewhere
      d1 = exp(z) / (1.0d0 + exp(z))
    end where

    x  = log(1.0d0 + exp(z))   ! φ itself
    d2 = d1 * (1.0d0 - d1)
    d3 = d2 * (1.0d0 - 2.0d0*d1)

    deallocate(mask)
  end subroutine softplus_derivs

end module act_derivs

    

    
module deriv_recursive
  use act_derivs
  use forward_pass
  implicit none
  private
  public :: compute_pre, derivatives_output

contains

  !------------------------------------------------------------------
  ! compute_pre:
  !   Builds Pre(:,:,l) = d x^{(l-1)} / d x for l=1..L
  !   Pre(:,:,1) = I_{n0}
  !   Pre(:,:,l+1) = (D^{(l)} W^{(l)}) @ Pre(:,:,l)
  !   Shapes are padded to (max_out, n0, L)
  !------------------------------------------------------------------
  subroutine compute_pre(weights, d1_list, n0, nout, nin, L, Pre)
    implicit none
    real(8), intent(in)  :: weights(:,:,:)
    real(8), intent(in)  :: d1_list(:,:)
    integer, intent(in)  :: n0, L
    integer, intent(in)  :: nout(:), nin(:)
    real(8), allocatable, intent(out) :: Pre(:,:,:)

    integer :: ll, ii, jj, aa, bb, max_out
    real(8), allocatable :: DlW(:,:)

    max_out = size(weights,1)
    allocate(Pre(max_out, n0, L+1))
    Pre = 0.0d0

    ! Pre(:,:,1) = I (size n0)
    do ii=1,n0
      Pre(ii,ii,1) = 1.0d0
    end do

    ! Recurrence for l = 1..L-1: Pre(:,:,l+1) = (D^l W^l) @ Pre(:,:,l)
    do ll = 1, L
      allocate(DlW(nout(ll), nin(ll)))
      DlW = 0.0d0

      ! DlW = diag(d1_list(1:nout(l), l)) * weights(1:nout(l),1:nin(l),l)
      do aa = 1, nout(ll)
        do bb = 1, nin(ll)
          DlW(aa,bb) = d1_list(aa,ll) * weights(aa,bb,ll)
        end do
      end do

      ! Pre(1:nout(l),:,l+1) = DlW @ Pre(1:nin(l),:,l)
      do aa = 1, nout(ll)
        do jj = 1, n0
          do bb = 1, nin(ll)
            Pre(aa,jj,ll+1) = Pre(aa,jj,ll+1) + DlW(aa,bb) * Pre(bb,jj,ll)
          end do
        end do
      end do

      deallocate(DlW)
    end do
  end subroutine compute_pre


  !------------------------------------------------------------------
  ! derivatives_output:
  !     Ml  = Wl @ Pre(:,:,ll)
  !     K   = einsum('ar,rij->aij',  Wl, H_prev)
  !     Q   = einsum('ar,rijk->aijk', Wl, T_prev)
  !     H   = d2 * (Ml ⊗ Ml) + d1 * K
  !     T   = d3 * Ml*Ml*Ml + d2 * (mixed 3 terms) + d1 * Q
  !
  ! Inputs:
  !   weights(max_out, max_in, L)  - padded weight tensor
  !   acts(L)                      - activation codes (unused here but kept for symmetry)
  !   x(n0)                        - input vector
  !   nout(L), nin(L)              - actual per-layer sizes
  !
  ! Outputs:
  !   H_out(nL, n0, n0), T_out(nL, n0, n0, n0)  (top layer only)
  !------------------------------------------------------------------    
subroutine derivatives_output(weights, biases, acts, x, nout, nin, L, J_out, H_out, T_out)
  use forward_pass
  implicit none
  ! Inputs
  real(8), intent(in)  :: weights(:,:,:)
  real(8), intent(in)  :: biases(:,:)   
  integer, intent(in)  :: acts(:)
  real(8), intent(in)  :: x(:)
  integer, intent(in)  :: L
  integer, intent(in)  :: nout(:), nin(:)
  ! Outputs
  real(8), allocatable, intent(out) :: J_out(:,:), H_out(:,:,:), T_out(:,:,:,:)

  ! Locals
  integer :: n0, ll, aa, ii, rr, jj, kk
  real(8), allocatable :: x_list(:,:), z_list(:,:), d1_list(:,:), d2_list(:,:), d3_list(:,:)
  real(8), allocatable :: Pre(:,:,:)
  real(8), allocatable :: Wl(:,:), d1(:), d2(:), d3(:), Ml(:,:)
  real(8), allocatable :: H_prev(:,:,:), T_prev(:,:,:,:)
  real(8), allocatable :: H_cur(:,:,:), T_cur(:,:,:,:)

  n0 = size(x)

  ! --- Forward pass with derivatives to get d1,d2,d3
  call forward_with_derivs(weights, biases, acts, x, x_list, z_list, d1_list, d2_list, d3_list, L)

  ! --- Compute Pre-Jacobians up to each layer: Pre(:,:,1)=I, Pre(:,:,l) = d x^{(l-1)}/dx
  call compute_pre(weights, d1_list, n0, nout, nin, L, Pre)

  ! --- Allocate outputs
  allocate(J_out(nout(L), n0));  J_out = 0.0d0
  allocate(H_out(nout(L), n0, n0))
  allocate(T_out(nout(L), n0, n0, n0))

  ! --- Initialize recursions
  allocate(H_prev(nin(1), n0, n0));  H_prev = 0.0d0
  allocate(T_prev(nin(1), n0, n0, n0)); T_prev = 0.0d0

  ! --- Loop over layers (layer recursion)
  do ll = 1, L
    ! Slice the active W^l and derivatives
    allocate(Wl(nout(ll), nin(ll)))
    Wl = weights(1:nout(ll),1:nin(ll),ll)

    allocate(d1(nout(ll))); d1 = d1_list(1:nout(ll),ll)
    allocate(d2(nout(ll))); d2 = d2_list(1:nout(ll),ll)
    allocate(d3(nout(ll))); d3 = d3_list(1:nout(ll),ll)

    ! Ml = Wl @ Pre(:,:,l) (size: nout(ll) x n0)
    allocate(Ml(nout(ll), n0)); Ml = 0.0d0
    do aa = 1, nout(ll)
      do jj = 1, n0
        do rr = 1, nin(ll)
          Ml(aa,jj) = Ml(aa,jj) + Wl(aa,rr) * Pre(rr,jj,ll)
        end do
      end do
    end do

    ! If this is the final layer, form the Jacobian: J = diag(d1^L) * Ml
    if (ll == L) then
      do aa = 1, nout(ll)
        do jj = 1, n0
          J_out(aa,jj) = d1(aa) * Ml(aa,jj)
        end do
      end do
    end if

    ! --- Contract with previous Hessian/Tensor: K = Wl @ H_prev, Q = Wl @ T_prev
    allocate(H_cur(nout(ll), n0, n0)); H_cur = 0.0d0
    allocate(T_cur(nout(ll), n0, n0, n0)); T_cur = 0.0d0

    ! --- Updates
    do aa = 1, nout(ll)
      do rr = 1, nin(ll)
        H_cur(aa,:,:)   = H_cur(aa,:,:)   + Wl(aa,rr) * H_prev(rr,:,:)
        T_cur(aa,:,:,:) = T_cur(aa,:,:,:) + Wl(aa,rr) * T_prev(rr,:,:,:)
      end do
    end do

    ! --- Third derivative update: T = d3*Ml^⊗3 + d2*sym(Ml⊗K) + d1*Q
    do aa = 1, nout(ll)
      do ii = 1, n0
        do jj = 1, n0
          do kk = 1, n0
            T_cur(aa,ii,jj,kk) = d3(aa)*Ml(aa,ii)*Ml(aa,jj)*Ml(aa,kk) &   ! Local cubic term
                               + d2(aa)*( H_cur(aa,ii,jj)*Ml(aa,kk)   &   ! Mixed terms (3 permutations) with d2
                                        + H_cur(aa,ii,kk)*Ml(aa,jj)   &
                                        + H_cur(aa,jj,kk)*Ml(aa,ii) ) &
                               + d1(aa)*T_cur(aa,ii,jj,kk)                ! Propagation term with d1 * Q
          end do
        end do
      end do
      
      ! --- Hessian update: H = d2 * (Ml ⊗ Ml) + d1 * K
      do ii = 1, n0
        do jj = 1, n0
          H_cur(aa,ii,jj) = d2(aa) * Ml(aa,ii) * Ml(aa,jj) + d1(aa) * H_cur(aa,ii,jj)
        end do
      end do
    end do

    ! Prepare next iteration
    deallocate(Ml, H_prev, T_prev)
    allocate(H_prev(nout(l), n0, n0))
    allocate(T_prev(nout(l), n0, n0, n0))
    H_prev = H_cur
    T_prev = T_cur

    ! Cleanup layer-local arrays
    deallocate(Wl, d1, d2, d3, H_cur, T_cur)
  end do

  ! --- Final outputs
  H_out(:,:,:) = H_prev
  T_out(:,:,:,:) = T_prev

  deallocate(x_list, z_list, d1_list, d2_list, d3_list, Pre, H_prev, T_prev)

end subroutine derivatives_output


end module

    
    
program test_driver

  use act_derivs
  use forward_pass
  use deriv_recursive
  
  implicit none
  
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: L = 2
  integer :: nout(L), nin(L), max_in, max_out, n0
  real(dp), allocatable :: weights(:,:,:)
  real(dp), allocatable :: biases(:,:)
  real(dp), allocatable :: J(:,:), H(:,:,:), T(:,:,:,:)
  real(dp), allocatable :: x(:)
  integer :: acts(L)
  integer :: ii
  integer :: runs = 1
  
  ! --- timing
  integer, parameter :: rk = kind ( 1.0D+00 )
  real( kind = rk ) :: start, finish, wtime

    ! --- Real weights from trained model_Psi
  real(dp), dimension(8, 2) :: W1
  real(dp), dimension(8) :: b1
  real(dp), dimension(1, 8) :: W2
  real(dp), dimension(1) :: b2

  W1 = reshape([ &
    1.2888870017946965e+00_dp, 1.5145195938264308e-01_dp,  &
    1.9320580926652693e+00_dp, 1.9918527939818185e+00_dp,  &
    4.8880605855934334e-01_dp, 9.2730949905271043e-01_dp,  &
    1.6293628219819198e+00_dp, 9.2914849523072829e-01_dp,  &
    1.2481604161743696e+00_dp, 1.4458479075812265e+00_dp,  &
    1.8912267332071093e+00_dp, 5.1533571319971078e-02_dp,  &
    3.6482090123405425e-01_dp, 4.5470015720466872e-01_dp,  &
    1.6829053220325052e+00_dp, 5.2489295386673540e-01_dp &
  ], [8, 2])

  b1 = [ 2.3965488542613889e-01_dp, -5.9100950617548098e-01_dp, -3.7865896179789643e-01_dp, -5.5185781774482068e-01_dp, 1.9949575683286414e-01_dp, 9.3814822552803712e-02_dp, -5.8229248969334158e-01_dp, -5.5133114124484950e-01_dp ]

  W2 = reshape([ &
    4.9257802500604847e-01_dp, 9.3675650073927319e-01_dp,  &
    4.1396657673520842e-01_dp, 6.9675046432186596e-02_dp,  &
    1.3964157427103915e-01_dp, 4.3397306939393776e-01_dp,  &
    8.2451670735190619e-01_dp, 3.5235516796881644e-01_dp &
  ], [1, 8])

  b2 = [ 0.0000000000000000e+00_dp ]

  ! --- Network sizes
  L = 2
  n0 = 8
  nin(1) = 2; nout(1) = 8
  nin(2) = 8; nout(2) = 1

  ! --- Compute derivatives
  start = wtime ( )
  do ii = 1, runs
      call derivatives_output(weights, biases, acts, x, nout, nin, L, J, H, T)
  end do
  finish = wtime ( ) - start
  print '("Time = ",f6.3," seconds.")', finish

  ! --- Sanity checks
  print *, 'Shapes: J(', size(J,1), ',', size(J,2), ')'
  print *, '         H(', size(H,1), ',', size(H,2), ',', size(H,3), ')'
  print *, '         T(', size(T,1), ',', size(T,2), ',', size(T,3), ',', size(T,4), ')'
  print *, 'Sample J(1,1)   =', J(1,1)
  print *, 'Sample H(1,1,1) =', H(1,1,1)
  print *, 'Sample T(1,1,1,1) =', T(1,1,1,1)
  read (*,*)

end program test_driver
  
    
    
    
    
function wtime ( )

!*****************************************************************************80
!
!! WTIME returns a reading of the wall clock time.
!
!  Discussion:
!
!    To get the elapsed wall clock time, call WTIME before and after a given
!    operation, and subtract the first reading from the second.
!
!    This function is meant to suggest the similar routines:
!
!      "omp_get_wtime ( )" in OpenMP,
!      "MPI_Wtime ( )" in MPI,
!      and "tic" and "toc" in MATLAB.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    27 April 2009
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Output, real ( kind = rk ) WTIME, the wall clock reading, in seconds.
!
  implicit none

  integer, parameter :: rk = kind ( 1.0D+00 )

  integer clock_max
  integer clock_rate
  integer clock_reading
  real ( kind = rk ) wtime

  call system_clock ( clock_reading, clock_rate, clock_max )

  wtime = real ( clock_reading, kind = rk ) &
        / real ( clock_rate, kind = rk )

  return
end