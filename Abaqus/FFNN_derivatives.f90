module act_derivs
  use precision
  implicit none
  private
  public :: tanh_derivs, sigmoid_derivs, softplus_derivs, linear_derivs, elu_derivs, softplus_sq_derivs
  
contains

  !-----------------------------------------------------------
  ! tanh: φ(z) = t, d1 = 1 - t^2
  !       d2 = -2 t (1 - t^2)
  !       d3 = -2 (1 - t^2) (1 - 3 t^2)
  !-----------------------------------------------------------
  subroutine tanh_derivs(z, x, d1, d2, d3)
    implicit none
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)
    real(dp), allocatable :: one_minus_t2(:)
    integer :: n

    n = size(z)
    allocate(one_minus_t2(n))

    x  = tanh(z)
    one_minus_t2 = 1.0_dp - x*x

    d1 = one_minus_t2
    d2 = -2.0_dp * x * one_minus_t2
    d3 = -2.0_dp * one_minus_t2 * (1.0_dp - 3.0_dp*x*x)

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
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)
    logical, allocatable :: pos(:)
    integer :: n

    n = size(z)
    allocate(pos(n))

    ! Stable computation of sigmoid:
    pos = (z >= 0.0_dp)
    x    = 0.0_dp
    where (pos)
      x = 1.0_dp / (1.0_dp + exp(-z))
    elsewhere
      x = exp(z) / (1.0_dp + exp(z))
    end where

    d1 = x * (1.0_dp - x)
    d2 = d1 * (1.0_dp - 2.0_dp*x)
    d3 = d1 * (1.0_dp - 6.0_dp*x + 6.0_dp*x*x)

    deallocate(pos)
  end subroutine sigmoid_derivs


  !-----------------------------------------------------------
  ! Softplus φ(z) = log(1+exp(z)), φ′ = s = sigmoid(z)
  !   d1 = s
  !   d2 = s(1-s)
  !   d3 = d2(1-2s)
  ! Numerically stable φ:
  !   φ(z) = max(z,0) + log(1 + exp(-abs(z)))
  !-----------------------------------------------------------
  subroutine softplus_derivs(z, x, d1, d2, d3)
    implicit none
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)
    logical, allocatable :: pos(:)
    integer :: n

    n = size(z)
    allocate(pos(n))

    ! Compute sigmoid s, reuse as d1
    pos = (z >= 0.0_dp)
    d1   = 0.0_dp
    where (pos)
      d1 = 1.0_dp / (1.0_dp + exp(-z))
      x = z + log(1.0_dp + exp(-z)) 
    elsewhere
      d1 = exp(z) / (1.0_dp + exp(z))
      x = log(1.0_dp + exp(z))
    end where

    d2 = d1 * (1.0_dp - d1)
    d3 = d2 * (1.0_dp - 2.0_dp*d1)

    deallocate(pos)
  end subroutine softplus_derivs


  !-----------------------------------------------------------
  ! Linear: φ(z) = z
  !   d1 = 1, d2 = 0, d3 = 0
  !-----------------------------------------------------------
  subroutine linear_derivs(z, x, d1, d2, d3)
    implicit none
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)

    x  = z
    d1 = 1.0_dp
    d2 = 0.0_dp
    d3 = 0.0_dp
  end subroutine linear_derivs


  !-----------------------------------------------------------
  ! ELU (α = 1): 
  !   φ(z) = z            , z > 0
  !   φ(z) = exp(z) - 1   , z ≤ 0
  !   d1 = { 1, exp(z) }, d2 = { 0, exp(z) }, d3 = { 0, exp(z) }
  !-----------------------------------------------------------
  subroutine elu_derivs(z, x, d1, d2, d3)
    implicit none
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)
    logical, allocatable :: pos(:)
    integer :: n
    n = size(z)
    allocate(pos(n))

    pos = (z > 0.0_dp)

    ! Initialize to ELU negative-branch (α = 1)
    x  = exp(z) - 1.0_dp
    d1 = exp(z)
    d2 = exp(z)
    d3 = exp(z)

    ! Overwrite where z > 0 with linear branch
    where (pos)
      x  = z
      d1 = 1.0_dp
      d2 = 0.0_dp
      d3 = 0.0_dp
    end where

    deallocate(pos)
  end subroutine elu_derivs


  !-----------------------------------------------------------
  ! Squared Softplus: φ(z) = p^2, where p = log(1 + exp(z))
  ! Let s = sigmoid(z) and t = s(1-s), u = 1 - 2s.
  !   d1 = 2 p s
  !   d2 = 2 ( s^2 + p t )
  !   d3 = 2 t ( 3 s + p u )
  ! Uses stable branches for s and p.
  !-----------------------------------------------------------
  subroutine softplus_sq_derivs(z, x, d1, d2, d3)
    implicit none
    real(dp), intent(in)  :: z(:)
    real(dp), intent(out) :: x(:), d1(:), d2(:), d3(:)
    integer :: n
    logical, allocatable :: pos(:)
    real(dp), allocatable :: s(:), p(:), t(:), u(:)

    n = size(z)
    allocate(pos(n), s(n), p(n), t(n), u(n))

    pos = (z >= 0.0_dp)

    ! Stable sigmoid and softplus
    where (pos)
      s = 1.0_dp / (1.0_dp + exp(-z))             ! sigmoid
      p = z + log(1.0_dp + exp(-z))               ! softplus: z + log(1+exp(-z))
    elsewhere
      s = exp(z) / (1.0_dp + exp(z))
      p = log(1.0_dp + exp(z))
     end where

    t = s * (1.0_dp - s)                          ! s'
    u = 1.0_dp - 2.0_dp * s

    x  = p * p
    d1 = 2.0_dp * p * s
    d2 = 2.0_dp * ( s*s + p * t )
    d3 = 2.0_dp * t * ( 3.0_dp*s + p * u )

    deallocate(pos, s, p, t, u)
  end subroutine softplus_sq_derivs

end module act_derivs
    
! ================================================================================================ !
    
module forward_pass
  use precision
  use act_derivs
  implicit none
  private
  public :: forward_with_derivs

contains

  !================================================================
  ! Forward pass: computes x^l, z^l, and φ′, φ″, φ‴ at each layer
  !================================================================
  subroutine forward_with_derivs(weights, biases, acts, nin, nout, x, x_list, z_list, d1_list, d2_list, d3_list, L)
    implicit none
    ! Inputs
    real(dp), intent(in)  :: weights(:,:,:)
    real(dp), intent(in)  :: biases(:,:)   
    integer, intent(in)  :: acts(:)
    real(dp), intent(in)  :: x(:)
    integer, intent(in)  :: nin(:), nout(:)
    integer, intent(in)  :: L

    ! Outputs
    real(dp), allocatable, intent(out) :: x_list(:,:), z_list(:,:), d1_list(:,:), d2_list(:,:), d3_list(:,:)

    ! Locals
    integer :: ll, n_in, n_out, max_in, max_out, max_all
    real(dp), allocatable :: z(:), d1(:), d2(:), d3(:), x_next(:)

    ! Find maximum layer width for allocation convenience
    max_out = size(weights,1)
    max_in  = size(weights,2)
    max_all = max(size(x), max(max_in, max_out))
    
    ! Allocate lists
    allocate(x_list(max_all,L+1))
    allocate(z_list(max_out,L))     ! z only needs max_out
    allocate(d1_list(max_out,L))
    allocate(d2_list(max_out,L))
    allocate(d3_list(max_out,L))

    ! Store input as x^0
    x_list(:,1) = 0.0_dp
    x_list(1:size(x),1) = x

    ! Loop over layers
    do ll = 1,L
       n_in  = nin(ll)
       n_out = nout(ll)

       allocate(z(n_out), d1(n_out), d2(n_out), d3(n_out), x_next(n_out))

       ! Linear transform: z^l = W^l x^{l-1} + b^l
       z = matmul(weights(1:n_out,1:n_in,ll), x_list(1:n_in,ll)) + biases(1:n_out,ll)

       ! Store preactivation
       z_list(:,ll) = 0.0_dp
       z_list(1:n_out,ll) = z

       select case(acts(ll))
       case(1)   ! tanh
          call tanh_derivs(z, x_next, d1, d2, d3)
       case(2)   ! sigmoid
          call sigmoid_derivs(z, x_next, d1, d2, d3)
       case(3)   ! softplus
          call softplus_derivs(z, x_next, d1, d2, d3)
       case(4)   ! linear
          call linear_derivs(z, x_next, d1, d2, d3)
       case(5)   ! ELU (alpha = 1)
          call elu_derivs(z, x_next, d1, d2, d3)
       case(6)   ! ELU (alpha = 1)
          call softplus_sq_derivs(z, x_next, d1, d2, d3)
       case default
          call linear_derivs(z, x_next, d1, d2, d3)
       end select

       ! Store results
       d1_list(:,ll) = 0.0_dp; d2_list(:,ll) = 0.0_dp; d3_list(:,ll) = 0.0_dp
       d1_list(1:n_out,ll) = d1
       d2_list(1:n_out,ll) = d2
       d3_list(1:n_out,ll) = d3

       x_list(:,ll+1) = 0.0_dp
       x_list(1:n_out,ll+1) = x_next

       ! Deallocate temporaries
       deallocate(z, d1, d2, d3, x_next)
    end do

  end subroutine forward_with_derivs

end module forward_pass

! ================================================================================================ !

module deriv_recursive
  use precision
  use act_derivs
  use forward_pass
  implicit none
  private
  public :: compute_pre, derivatives_output, derivatives_expanded_output

  contains

  !================================================================
  ! compute_pre:
  !   Builds Pre(:,:,l) = d x^{(l-1)} / d x for l=1..L
  !   Pre(:,:,1) = I_{n0}
  !   Pre(:,:,l+1) = (D^{(l)} W^{(l)}) @ Pre(:,:,l)
  !   Shapes are padded to (max_all, n0, L)
  !================================================================
  subroutine compute_pre(weights, d1_list, n0, nout, nin, L, Pre)
    implicit none
    real(dp), intent(in)  :: weights(:,:,:)
    real(dp), intent(in)  :: d1_list(:,:)
    integer, intent(in)  :: n0, L
    integer, intent(in)  :: nout(:), nin(:)
    real(dp), allocatable, intent(out) :: Pre(:,:,:)

    integer :: ll, ii, jj, aa, bb, max_nin, max_nout, max_all
    real(dp), allocatable :: DlW(:,:)

    max_nin  = maxval(nin(1:L))
    max_nout = maxval(nout(1:L))
    max_all  = max(n0, max(max_nin, max_nout))
    
    allocate(Pre(max_all, n0, L+1))
    Pre = 0.0_dp

    ! Pre(:,:,1) = I (size n0)
    do ii=1,n0
      Pre(ii,ii,1) = 1.0_dp
    end do

    ! Recurrence for l = 1..L-1: Pre(:,:,l+1) = (D^l W^l) @ Pre(:,:,l)
    do ll = 1, L
        
      allocate(DlW(nout(ll), nin(ll)))
      DlW = 0.0_dp

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


  !================================================================
  ! derivatives_output:
  !     Ml  = Wl @ Pre(:,:,ll)
  !     K   = einsum('ar,rij->aij',  Wl, H_prev)
  !     Q   = einsum('ar,rijk->aijk', Wl, T_prev)
  !     H   = d2 * (Ml ⊗ Ml) + d1 * K
  !     T   = d3 * Ml*Ml*Ml + d2 * (mixed 3 terms) + d1 * Q
  !
  ! Inputs:
  !   weights(max_out, max_in, L)  - padded weight tensor
  !   acts(L)                      - activation codes
  !   x(n0)                        - input vector
  !   nout(L), nin(L)              - actual per-layer sizes
  !
  ! Outputs:
  !   y_out(nL), J_out(nL, n0), H_out(nL, n0, n0), T_out(nL, n0, n0, n0)
  !================================================================    
  subroutine derivatives_output(weights, biases, acts, x, nout, nin, L, &
                                want_hessian, want_third,               &
                                J_out, H_out, T_out, y_out)
    use forward_pass
    implicit none
    ! Inputs
    real(dp), intent(in)  :: weights(:,:,:)
    real(dp), intent(in)  :: biases(:,:)
    integer, intent(in)  :: acts(:)
    real(dp), intent(in)  :: x(:)
    integer, intent(in)  :: L
    integer, intent(in)  :: nout(:), nin(:)
    logical, intent(in)  :: want_hessian, want_third
    ! Outputs
    real(dp), allocatable, intent(out) :: J_out(:,:), H_out(:,:,:), T_out(:,:,:,:)     ! derivatives
    real(dp), allocatable, intent(out) :: y_out(:)     ! final activations


    ! Locals
    integer :: n0, ll, aa, ii, rr, jj, kk
    logical :: need_H, need_T
    real(dp), allocatable :: x_list(:,:), z_list(:,:), d1_list(:,:), d2_list(:,:), d3_list(:,:)
    real(dp), allocatable :: Pre(:,:,:)
    real(dp), allocatable :: Wl(:,:), d1(:), d2(:), d3(:), Ml(:,:)
    real(dp), allocatable :: H_prev(:,:,:), T_prev(:,:,:,:)
    real(dp), allocatable :: H_cur(:,:,:), T_cur(:,:,:,:)

    n0 = size(x)
    need_T = want_third
    need_H = want_hessian .or. need_T   ! third derivative needs H-like intermediates

    ! --- Forward pass with derivatives to get d1,d2,d3
    call forward_with_derivs(weights, biases, acts, nin, nout, x, x_list, z_list, d1_list, d2_list, d3_list, L)
    
    ! --- Network output (final activations)
    allocate(y_out(nout(L)))
    y_out = x_list(1:nout(L), L+1)

    ! --- Compute Pre-Jacobians up to each layer: Pre(:,:,1)=I, Pre(:,:,l) = d x^{(l-1)}/dx
    call compute_pre(weights, d1_list, n0, nout, nin, L, Pre)

    ! --- Allocate outputs
    allocate(J_out(nout(L), n0));  J_out = 0.0_dp
    if (want_hessian) then
      allocate(H_out(nout(L), n0, n0))
    else
      allocate(H_out(0,0,0))   ! zero-sized when not requested
    end if
    if (want_third) then
      allocate(T_out(nout(L), n0, n0, n0))
    else
      allocate(T_out(0,0,0,0)) ! zero-sized when not requested
    end if

    ! --- Initialize recursions only if needed
    if (need_H) then
      allocate(H_prev(nin(1), n0, n0));  H_prev = 0.0_dp
    end if
    if (need_T) then
      allocate(T_prev(nin(1), n0, n0, n0)); T_prev = 0.0_dp
    end if

    ! --- Loop over layers (layer recursion)
    do ll = 1, L
      ! Slice the active W^l and derivatives
      allocate(Wl(nout(ll), nin(ll)))
      Wl = weights(1:nout(ll),1:nin(ll),ll)

      allocate(d1(nout(ll))); d1 = d1_list(1:nout(ll),ll)
      allocate(d2(nout(ll))); d2 = d2_list(1:nout(ll),ll)
      allocate(d3(nout(ll))); d3 = d3_list(1:nout(ll),ll)

      ! Ml = Wl @ Pre(:,:,l) (size: nout(ll) x n0)
      allocate(Ml(nout(ll), n0)); Ml = 0.0_dp
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

      
      ! If neither Hessian nor third are needed, skip all below
      if (need_H .or. need_T) then

        if (need_H) then
          allocate(H_cur(nout(ll), n0, n0)); H_cur = 0.0_dp
          ! Contract with previous Hessian: K = Wl @ H_prev
          !do aa = 1, nout(ll)
          !  do rr = 1, nin(ll)
          !    H_cur(aa,:,:) = H_cur(aa,:,:) + Wl(aa,rr) * H_prev(rr,:,:)
          !  end do
          !end do
          do jj = 1, n0
            H_cur(:,:,jj) = matmul(Wl, H_prev(:,:,jj))
          end do
        end if
        
        if (need_T) then
          allocate(T_cur(nout(ll), n0, n0, n0)); T_cur = 0.0_dp
          ! Contract with previous third derivative: Q = Wl @ T_prev
          !do aa = 1, nout(ll)
          !  do rr = 1, nin(ll)
          !    T_cur(aa,:,:,:) = T_cur(aa,:,:,:) + Wl(aa,rr) * T_prev(rr,:,:,:)
          !  end do
          !end do
          do kk = 1, n0
            do jj = 1, n0
              T_cur(:,:,jj,kk) = matmul(Wl, T_prev(:,:,jj,kk))
            end do
          end do
        end if

        ! Third derivative update: T = d3*Ml^⊗3 + d2*sym(Ml⊗K) + d1*Q
        if (need_T) then
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
          end do
        end if
      
        ! --- Hessian update: H = d2 * (Ml ⊗ Ml) + d1 * K
        if (need_H) then
          do aa = 1, nout(ll)
            do ii = 1, n0
              do jj = 1, n0
                H_cur(aa,ii,jj) = d2(aa) * Ml(aa,ii) * Ml(aa,jj) + d1(aa) * H_cur(aa,ii,jj)
              end do
            end do
          end do
        end if
        
        ! Prepare next iteration only for the quantities we keep
        if (need_H) then
          if (allocated(H_prev)) deallocate(H_prev)
          allocate(H_prev(nout(ll), n0, n0))
          H_prev = H_cur
          deallocate(H_cur)
        end if
        if (need_T) then
          if (allocated(T_prev)) deallocate(T_prev)
          allocate(T_prev(nout(ll), n0, n0, n0))
          T_prev = T_cur
          deallocate(T_cur)
        end if

      end if  ! need_H or need_T

      ! Cleanup layer-local arrays
      deallocate(Wl, d1, d2, d3, Ml)
    end do

    ! --- Final outputs (only assign when requested)
    if (want_hessian) then
      H_out(:,:,:) = H_prev
    end if
    if (want_third) then
      T_out(:,:,:,:) = T_prev
    end if

    ! Final cleanup
    deallocate(x_list, z_list, d1_list, d2_list, d3_list, Pre)
    if (allocated(H_prev)) deallocate(H_prev)
    if (allocated(T_prev)) deallocate(T_prev)

  end subroutine derivatives_output


  !================================================================
  ! derivatives_expanded_output:
  !   Computes Jacobian/Hessian/Third derivative using the expanded (post-Jacobian) formulation.
  !   Reuses the forward pass derivatives and Pre tensors but
  !   accumulates contributions by propagating layer-local terms
  !   towards the output via Post matrices.
  !================================================================
  subroutine derivatives_expanded_output(weights, biases, acts, x, nout, nin, L, &
                                         want_hessian, want_third,               &
                                         J_out, H_out, T_out, y_out)
    use forward_pass
    implicit none
    ! Inputs
    real(dp), intent(in)  :: weights(:,:,:)
    real(dp), intent(in)  :: biases(:,:)
    integer, intent(in)  :: acts(:)
    real(dp), intent(in)  :: x(:)
    integer, intent(in)  :: nout(:), nin(:), L
    logical, intent(in)  :: want_hessian, want_third
    ! Outputs
    real(dp), allocatable, intent(out) :: J_out(:,:), H_out(:,:,:), T_out(:,:,:,:)
    real(dp), allocatable, intent(out) :: y_out(:)

    ! Locals
    integer :: n0, nL, max_nout, max_nin, ll, aa, bb, jj, ii, kk, mm, ss, pp
    real(dp), allocatable :: x_list(:,:), z_list(:,:), d1_list(:,:), d2_list(:,:), d3_list(:,:)
    real(dp), allocatable :: Pre(:,:,:)
    real(dp), allocatable :: M_list(:,:,:)
    real(dp), allocatable :: Post(:,:,:)
    real(dp), allocatable :: A_mats(:,:,:)
    real(dp), allocatable :: K(:,:,:)
    real(dp), allocatable :: J_pair(:,:,:,:)
    real(dp), allocatable :: C_mat(:,:,:,:)
    real(dp) :: coeff, val_i, val_j, val_k, tmp

    n0 = size(x)
    nL = nout(L)
    max_nout = maxval(nout(1:L))
    max_nin  = maxval(nin(1:L))

    ! Forward pass to gather activation derivatives
    call forward_with_derivs(weights, biases, acts, nin, nout, x, x_list, z_list, d1_list, d2_list, d3_list, L)
    allocate(y_out(nL))
    y_out = x_list(1:nL, L+1)

    ! Build Pre tensors (derivatives of intermediate states w.r.t. inputs)
    call compute_pre(weights, d1_list, n0, nout, nin, L, Pre)

    ! Allocate outputs
    allocate(J_out(nL, n0)); J_out = 0.0_dp
    if (want_hessian) then
      allocate(H_out(nL, n0, n0)); H_out = 0.0_dp
    else
      allocate(H_out(0,0,0))
    end if
    if (want_third) then
      allocate(T_out(nL, n0, n0, n0)); T_out = 0.0_dp
    else
      allocate(T_out(0,0,0,0))
    end if

    ! Precompute A matrices (diag(d1) * W) and forward M tensors W @ Pre
    allocate(A_mats(max_nout, max_nin, L)); A_mats = 0.0_dp
    allocate(M_list(max_nout, n0, L));       M_list = 0.0_dp
    do ll = 1, L
      do aa = 1, nout(ll)
        do bb = 1, nin(ll)
          A_mats(aa, bb, ll) = d1_list(aa, ll) * weights(aa, bb, ll)
        end do
      end do
      do aa = 1, nout(ll)
        do jj = 1, n0
          do bb = 1, nin(ll)
            M_list(aa, jj, ll) = M_list(aa, jj, ll) + weights(aa, bb, ll) * Pre(bb, jj, ll)
          end do
        end do
      end do
    end do

    ! Build Post matrices: Post(:, :, l) = J(l -> L)
    allocate(Post(nL, max_nout, L)); Post = 0.0_dp
    do aa = 1, nL
      Post(aa, aa, L) = 1.0_dp
    end do
    do ll = L-1, 1, -1
      do pp = 1, nL
        do aa = 1, nout(ll)
          tmp = 0.0_dp
          do bb = 1, nout(ll+1)
            tmp = tmp + Post(pp, bb, ll+1) * A_mats(bb, aa, ll+1)
          end do
          Post(pp, aa, ll) = tmp
        end do
      end do
    end do

    ! Jacobian: final layer only
    do aa = 1, nL
      do jj = 1, n0
        J_out(aa, jj) = d1_list(aa, L) * M_list(aa, jj, L)
      end do
    end do

    ! Hessian accumulation via expanded form
    if (want_hessian) then
      do ll = 1, L
        do pp = 1, nL
          do aa = 1, nout(ll)
            coeff = Post(pp, aa, ll) * d2_list(aa, ll)
            if (coeff /= 0.0_dp) then
              do ii = 1, n0
                val_i = coeff * M_list(aa, ii, ll)
                do jj = 1, n0
                  H_out(pp, ii, jj) = H_out(pp, ii, jj) + val_i * M_list(aa, jj, ll)
                end do
              end do
            end if
          end do
        end do
      end do
    end if

    ! Preparations specific to the third derivative
    if (want_third .and. L > 1) then
      allocate(J_pair(max_nout, max_nout, L, L)); J_pair = 0.0_dp
      do ss = 1, L-1
        do aa = 1, nout(ss)
          J_pair(aa, aa, ss, ss) = 1.0_dp
        end do
        do ll = ss+1, L-1
          do aa = 1, nout(ll)
            do bb = 1, nout(ss)
              tmp = 0.0_dp
              do mm = 1, nout(ll-1)
                tmp = tmp + A_mats(aa, mm, ll) * J_pair(mm, bb, ss, ll-1)
              end do
              J_pair(aa, bb, ss, ll) = tmp
            end do
          end do
        end do
      end do

      allocate(C_mat(max_nout, max_nout, L, L)); C_mat = 0.0_dp
      do ll = 2, L
        do ss = 1, ll-1
          if (ss == ll-1) then
            do aa = 1, nout(ll)
              do bb = 1, nout(ss)
                C_mat(aa, bb, ll, ss) = weights(aa, bb, ll)
              end do
            end do
          else
            do aa = 1, nout(ll)
              do bb = 1, nout(ss)
                tmp = 0.0_dp
                do mm = 1, nout(ll-1)
                  tmp = tmp + weights(aa, mm, ll) * J_pair(mm, bb, ss, ll-1)
                end do
                C_mat(aa, bb, ll, ss) = tmp
              end do
            end do
          end if
        end do
      end do
    end if

    ! Third derivative accumulation
    if (want_third) then
      do ll = 1, L
        ! Local cubic contribution
        do pp = 1, nL
          do aa = 1, nout(ll)
            coeff = Post(pp, aa, ll) * d3_list(aa, ll)
            if (coeff /= 0.0_dp) then
              do ii = 1, n0
                val_i = coeff * M_list(aa, ii, ll)
                do jj = 1, n0
                  val_j = val_i * M_list(aa, jj, ll)
                  do kk = 1, n0
                    T_out(pp, ii, jj, kk) = T_out(pp, ii, jj, kk) + val_j * M_list(aa, kk, ll)
                  end do
                end do
              end do
            end if
          end do
        end do

        if (ll >= 2) then
          allocate(K(nout(ll), n0, n0)); K = 0.0_dp
          do ss = 1, ll-1
            do aa = 1, nout(ll)
              do bb = 1, nout(ss)
                coeff = C_mat(aa, bb, ll, ss) * d2_list(bb, ss)
                if (coeff /= 0.0_dp) then
                  do ii = 1, n0
                    val_i = coeff * M_list(bb, ii, ss)
                    do jj = 1, n0
                      K(aa, ii, jj) = K(aa, ii, jj) + val_i * M_list(bb, jj, ss)
                    end do
                  end do
                end if
              end do
            end do
          end do

          do pp = 1, nL
            do aa = 1, nout(ll)
              coeff = Post(pp, aa, ll) * d2_list(aa, ll)
              if (coeff /= 0.0_dp) then
                do ii = 1, n0
                  val_i = M_list(aa, ii, ll)
                  do jj = 1, n0
                    val_j = M_list(aa, jj, ll)
                    do kk = 1, n0
                      val_k = M_list(aa, kk, ll)
                      tmp = val_i * K(aa, jj, kk) + val_j * K(aa, ii, kk) + val_k * K(aa, ii, jj)
                      T_out(pp, ii, jj, kk) = T_out(pp, ii, jj, kk) + coeff * tmp
                    end do
                  end do
                end do
              end if
            end do
          end do
          deallocate(K)
        end if
      end do
    end if

    ! Cleanup
    deallocate(x_list, z_list, d1_list, d2_list, d3_list)
    deallocate(Pre, A_mats, M_list, Post)
    if (allocated(J_pair)) deallocate(J_pair)
    if (allocated(C_mat))  deallocate(C_mat)

  end subroutine derivatives_expanded_output


end module deriv_recursive