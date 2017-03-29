module constant!{{{
    implicit none
    integer(kind=4), parameter :: CREATION = 1
    integer(kind=4), parameter :: ANNIHILATION = 0
end module
!}}}

module countone!{{{
    implicit none
    private
    public :: numone, multi_numone
contains
    integer(kind=4) function numone(i, pos0, pos1)!{{{
        implicit none
        integer(kind=4), intent(in) :: i, pos0, pos1
        integer(kind=4) :: p0, p1, p
        if (pos0 < pos1) then
            p0 = pos0 + 1
            p1 = pos1 - 1
        else
            p0 = pos1 + 1
            p1 = pos0 - 1
        end if
    
        numone = 0
        if ((p1 - p0) < 0) then
            return
        else 
            do p=p0, p1, 1
                if (btest(i, p)) then
                    numone = numone + 1
                end if
            end do
        end if
        return
    end function numone
    !}}}

    subroutine multi_numone(ints, n, pos0, pos1, ones)!{{{
        implicit none
        integer(kind=4), intent(in) :: n, pos0, pos1, ints(0:n-1)
        integer(kind=4), intent(out) :: ones(0:n-1)
        integer(kind=4) :: i
        
        !$omp parallel if(n >= 10000) private(i) &
        !$omp& shared(n, pos0, pos1, ints, ones)
        !$omp do
        do i=0, n-1, 1
            ones(i) = numone(ints(i), pos0, pos1)
        end do
        !$omp end do
        !$omp end parallel
        return
    end subroutine multi_numone
    !}}}

end module
!}}}

module search!{{{
    implicit none
    private
    public :: bisearch, multi_bisearch
contains
    integer(kind=4) function bisearch(key, array, n)!{{{
        implicit none
        integer(kind=4), intent(in) :: key, n, array(0:n-1)
        integer(kind=4) :: low, high, mid, buff
        low = 0
        high = n-1
        do while(low<=high)
            mid = (low + high) / 2
            buff = array(mid)
            if (key == buff) then
                bisearch = mid
                return
            else if (key > buff) then
                low = mid + 1
            else
                high = mid - 1
            end if
        end do
        bisearch = -1
        return
    end function bisearch
    !}}}
    
    subroutine multi_bisearch(keys, array, keynum, n, poses)!{{{
        implicit none
        integer(kind=4), intent(in) :: keynum, n, keys(0:keynum-1), array(0:n-1)
        integer(kind=4), intent(out) :: poses(0:keynum-1)
        integer(kind=4) :: i

        !$omp parallel if(keynum >= 10000) private(i) &
        !$omp& shared(keynum, poses, keys, array, n)
        !$omp do
        do i=0, keynum-1, 1
            poses(i) = bisearch(keys(i), array, n)
        end do
        !$omp end do
        !$omp end parallel
        return
    end subroutine multi_bisearch
    !}}}

end module
!}}}

module matrepr!{{{
    implicit none
    private
    public :: hopping, hubbard, pairing, aoc
contains
    subroutine hopping(cindex, aindex, base, n, row, col, elmts)!{{{
        use constant
        use countone
        use search, only : bisearch
        implicit none
        integer(kind=4), intent(in) :: cindex, aindex, n, base(0:n-1)
        integer(kind=4), intent(out) :: row(0:n-1), col(0:n-1), elmts(0:n-1)
        integer(kind=4) :: ket, bra, i
        row = 0
        col = 0
        elmts = 0
    
        if (cindex == aindex) then
            !$omp parallel if(n>=10000) private(i, ket) &
            !$omp& shared(n, base, cindex, row, col, elmts)
            !$omp do
            do i=0, n-1, 1
                ket = base(i)
                if (btest(ket, cindex)) then
                    row(i) = i
                    col(i) = i
                    elmts(i) = 1
                end if
            end do
            !$omp end do
            !$omp end parallel
        else
            !$omp parallel if(n>=10000) private(i, ket, bra) &
            !$omp& shared(n, base, cindex, aindex, row, col, elmts)
            !$omp do
            do i=0, n-1, 1
                ket = base(i)
                if (.not. btest(ket, cindex) .and. btest(ket, aindex)) then
                    bra = ibset(ibclr(ket, aindex), cindex)
                    row(i) = bisearch(bra, base, n)
                    col(i) = i
                    elmts(i) = (-1) ** numone(ket, cindex, aindex)
                end if
            end do
            !$omp end do
            !$omp end parallel
        end if
        return
    end subroutine hopping
    !}}}
    
    subroutine hubbard(index0, index1, base, n, row, col, elmts)!{{{
        implicit none
        integer(kind=4), intent(in) :: index0, index1, n, base(0:n-1)
        integer(kind=4), intent(out) :: row(0:n-1), col(0:n-1), elmts(0:n-1)
        integer(kind=4) :: ket, i
        row = 0
        col = 0
        elmts = 0
    
        !$omp parallel if(n>=10000) private(i, ket) &
        !$omp& shared(n, base, index0, index1, row, col, elmts)
        !$omp do
        do i=0, n-1, 1
            ket = base(i)
            if (btest(ket, index0) .and. btest(ket, index1)) then
                row(i) = i
                col(i) = i
                elmts(i) = 1
            end if
        end do
        !$omp end do
        !$omp end parallel
        return
    end subroutine hubbard
    !}}}
    
    subroutine pairing(index0, index1, otype, base, n, row, col, elmts)!{{{
        use constant
        use countone
        use search, only : bisearch
        implicit none
        integer(kind=4), intent(in) :: index0, index1, otype, n, base(0:n-1)
        integer(kind=4), intent(out) :: row(0:n-1), col(0:n-1), elmts(0:n-1)
        integer(kind=4) :: ket, bra, i, extra
        row = 0
        col = 0
        elmts = 0
        extra = 0

        if (index0 == index1) then
            return
        end if

        if (otype == CREATION) then
            if (index0 > index1) then
                extra = 1
            end if

            !$omp parallel if(n>=10000) private(i, ket, bra) &
            !$omp& shared(n, base, index0, index1, row, col, elmts, extra)
            !$omp do
            do i=0, n-1, 1
                ket = base(i)
                if (.not. (btest(ket, index0) .or. btest(ket, index1))) then
                    bra = ibset(ibset(ket, index1), index0)
                    row(i) = bisearch(bra, base, n)
                    col(i) = i
                    elmts(i) = (-1) ** (numone(ket, index0, index1) + extra)
                end if
            end do
            !$omp end do
            !$omp end parallel
        else if (otype == ANNIHILATION) then
            if (index0 < index1) then
                extra = 1
            end if

            !$omp parallel if(n>=10000) private(i,ket, bra) &
            !$omp& shared(n, base, index0, index1, row, col, elmts, extra)
            !$omp do
            do i=0, n-1, 1
                ket = base(i)
                if (btest(ket, index0) .and. btest(ket, index1)) then
                    bra = ibclr(ibclr(ket, index1), index0)
                    row(i) = bisearch(bra, base, n)
                    col(i) = i
                    elmts(i) = (-1) ** (numone(ket, index0, index1) + extra)
                end if
            end do
            !$omp end do
            !$omp end parallel
        end if
        return
    end subroutine pairing
    !}}}
    
    subroutine aoc(ondex, otype, lbase, ln, rbase, rn, row, col, elmts)!{{{
        use constant
        use countone
        use search, only : bisearch
        implicit none
        integer(kind=4), intent(in) :: ondex, otype, ln, rn
        integer(kind=4), intent(in) :: lbase(0:ln-1), rbase(0:rn-1)
        integer(kind=4), intent(out) :: row(0:rn-1), col(0:rn-1), elmts(0:rn-1)
        integer(kind=4) :: ket, bra, i
        row = 0
        col = 0
        elmts = 0
    
        if (otype == CREATION) then
            !$omp parallel if(rn>=10000) private(i, ket, bra) &
            !$omp& shared(rn, rbase, ondex, ln, lbase, row, col, elmts)
            !$omp do
            do i=0, rn-1, 1
                ket = rbase(i)
                if (.not. btest(ket, ondex)) then
                    bra = ibset(ket, ondex)
                    row(i) = bisearch(bra, lbase, ln)
                    col(i) = i
                    elmts(i) = (-1) ** numone(ket, -1, ondex)
                end if
            end do
            !$omp end do
            !$omp end parallel
        else if (otype == ANNIHILATION) then
            !$omp parallel if(rn>=10000) private(i, ket, bra) &
            !$omp& shared(rn, rbase, ondex, ln, lbase, row, col, elmts)
            !$omp do
            do i=0, rn-1, 1
                ket = rbase(i)
                if (btest(ket, ondex)) then
                    bra = ibclr(ket, ondex)
                    row(i) = bisearch(bra, lbase, ln)
                    col(i) = i
                    elmts(i) = (-1) ** numone(ket, -1, ondex)
                end if
            end do
            !$omp end do
            !$omp end parallel
        end if
        return
    end subroutine aoc
    !}}}
end module matrepr
!}}}
