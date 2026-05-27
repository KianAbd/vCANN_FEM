!dec$ freeform 

module umat_statev_utils
    
  implicit none
  
contains

  ! ===================================================================
  ! Pack S_ra_bar, Q_ra, tau into STATEV (Voigt order: 11,22,33,12,13,23)
  ! ===================================================================
  subroutine set_statevs(S_ra_bar, Q_ra, tau, STATEV, numMaxwell, numTens, NSTATV)
    implicit none
    integer,           intent(in)    :: numMaxwell, numTens, NSTATV
    double precision,  intent(in)    :: S_ra_bar(3,3,numMaxwell,numTens)
    double precision,  intent(in)    :: Q_ra(3,3,numMaxwell,numTens)
    double precision,  intent(in)    :: tau(numMaxwell,numTens)
    double precision,  intent(inout) :: STATEV(NSTATV)

    integer, parameter :: nSym = 6
    integer :: iVoigt(nSym), jVoigt(nSym)
    integer :: ii, jj, k, offset

    data iVoigt /1,2,3,1,1,2/
    data jVoigt /1,2,3,2,3,3/

    offset = 0
    do ii = 1, numTens
      do jj = 1, numMaxwell

        ! S_ra_bar (6)
        do k = 1, nSym
          offset = offset + 1
          STATEV(offset) = S_ra_bar(iVoigt(k), jVoigt(k), jj, ii)
        end do

        ! Q_ra (6)
        do k = 1, nSym
          offset = offset + 1
          STATEV(offset) = Q_ra(iVoigt(k), jVoigt(k), jj, ii)
        end do

        ! tau (1)
        offset = offset + 1
        STATEV(offset) = tau(jj, ii)

      end do
    end do
  end subroutine set_statevs

  ! ===================================================================
  ! Unpack STATEV back into S_ra_bar, Q_ra, tau (reconstruct symmetry)
  ! ===================================================================
  subroutine get_statevs(STATEV, S_ra_bar, Q_ra, tau, numMaxwell, numTens, NSTATV)
    implicit none
    integer,           intent(in)  :: numMaxwell, numTens, NSTATV
    double precision,  intent(in)  :: STATEV(NSTATV)
    double precision,  intent(out) :: S_ra_bar(3,3,numMaxwell,numTens)
    double precision,  intent(out) :: Q_ra(3,3,numMaxwell,numTens)
    double precision,  intent(out) :: tau(numMaxwell,numTens)

    integer, parameter :: nSym = 6
    integer :: iVoigt(nSym), jVoigt(nSym)
    integer :: ii, jj, k, offset, i, j
    double precision :: val

    data iVoigt /1,2,3,1,1,2/
    data jVoigt /1,2,3,2,3,3/

    S_ra_bar = 0.0d0
    Q_ra     = 0.0d0
    tau      = 0.0d0

    offset = 0
    do ii = 1, numTens
      do jj = 1, numMaxwell

        ! S_ra_bar (6) -> fill symmetric entries
        do k = 1, nSym
          offset = offset + 1
          i = iVoigt(k);  j = jVoigt(k)
          val = STATEV(offset)
          S_ra_bar(i,j,jj,ii) = val
          S_ra_bar(j,i,jj,ii) = val
        end do

        ! Q_ra (6) -> fill symmetric entries
        do k = 1, nSym
          offset = offset + 1
          i = iVoigt(k);  j = jVoigt(k)
          val = STATEV(offset)
          Q_ra(i,j,jj,ii) = val
          Q_ra(j,i,jj,ii) = val
        end do

        ! tau (1)
        offset = offset + 1
        tau(jj,ii) = STATEV(offset)

      end do
    end do
  end subroutine get_statevs

  ! Optional helper to compute expected NSTATV (for setup/checks)
  pure integer function expected_nstatv(numMaxwell, numTens)
    implicit none
    integer, intent(in) :: numMaxwell, numTens
    expected_nstatv = (6 + 6 + 1) * numMaxwell * numTens
  end function expected_nstatv

end module umat_statev_utils
