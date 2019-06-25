program main
use ovf

    type(ovf_file)      :: file
    type(ovf_segment)   :: segment
    integer             :: success
    real(kind=4), allocatable :: array_4(:,:)
    real(kind=8), allocatable :: array_8(:,:)

    ! Initialize segment
    call segment%initialize()

    ! Nonexistent file
    call file%open_file("nonexistent.ovf")
    success = file%read_segment_header(segment)
    if ( success == OVF_ERROR) then
        write (*,*) "Found      = ", file%found
        write (*,*) "is_ovf     = ", file%is_ovf
        write (*,*) "n_segments = ", file%n_segments
    else
        write (*,*) "test read_segment_header on \'nonexistent.ovf\' should have returned OVF_ERROR... Message: ", &
            file%latest_message
        STOP 1
    endif
    ! Close file
    if ( file%close_file() == OVF_OK) then
        write (*,*) "nonexistent file closed"
    else
        write (*,*) "test close_file on \'nonexistent.ovf\' did not work."
        STOP 1
    endif

    ! Write a file
    call file%open_file("testfile_f.ovf")
    segment%ValueDim = 3
    segment%N_Cells = [ 2, 2, 1 ]
    segment%N = product(segment%N_Cells)

    allocate( array_4(3, segment%N) )
    array_4 = 0
    array_4(:,1) = [ 6.0, 7.0, 8.0 ]
    array_4(:,2) = [ 5.0, 4.0, 3.0 ]

    success = file%write_segment(segment, array_4, OVF_FORMAT_TEXT)
    if ( success == OVF_OK) then
        write (*,*) "test write_segment succeeded."
        ! write (*,*) "n_cells = ", segment%N_Cells
        ! write (*,*) "n_total = ", segment%N
    else
        write (*,*) "test write_segment did not work. Message: ", file%latest_message
        STOP 1
    endif

    ! Append to a file
    array_4(:,:) = 3*array_4(:,:)

    segment%Title = "fortran append test"
    success = file%append_segment(segment, array_4, OVF_FORMAT_TEXT)
    if ( success == OVF_OK) then
        write (*,*) "test append_segment succeeded."
        ! write (*,*) "n_cells = ", segment%N_Cells
        ! write (*,*) "n_total = ", segment%N
    else
        write (*,*) "test append_segment did not work. Message: ", file%latest_message
        STOP 1
    endif

    ! Read back in from file
    success = file%read_segment_header(segment)
    if( success == OVF_OK ) then
        write (*,*) "test read_segment_header:"
        write (*,*) "   n_cells = ", segment%N_Cells
        write (*,*) "   n_total = ", segment%N
    else
        write (*,*) "test read_segment_header did not work. Message: ", file%latest_message
        STOP 1
    endif

    success = file%read_segment_data(segment, array_8)
    if( success == OVF_OK ) then
        write (*,*) "test read_segment_data (index 1):"
        write (*,*) "   array_8(:,2) = ", array_8(:,2)
    else
        write (*,*) "test read_segment_data on array_8 did not work. Message: ", file%latest_message
        STOP 1
    endif

    success = file%read_segment_data(segment, array_8, 2)
    if( success == OVF_OK ) then
        write (*,*) "test read_segment_data (index 2):"
        write (*,*) "   array_8(:,2) = ", array_8(:,2)
    else
        write (*,*) "test read_segment_data on array_8 did not work. Message: ", file%latest_message
        STOP 1
    endif

    ! Close file
    if( file%close_file() == OVF_OK ) then
        write (*,*) "test file closed"
    else
        write (*,*) "test close_file on \'testfile_f.ovf\' did not work."
        STOP 1
    endif


end program main