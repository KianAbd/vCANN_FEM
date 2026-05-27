module precision
    implicit none
    public
    integer, parameter :: dp = selected_real_kind(15, 307) ! Double precision
end module