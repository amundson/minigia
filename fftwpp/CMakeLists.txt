add_library(fftwpp
    fftw++.cc
    fftw++.h
    mpifftw++.cc
    mpifftw++.h
    mpitranspose.cc
    mpitranspose.h
    mpigroup.h
    mpiutils.h
    utils.h
    timing.h)
target_link_libraries(fftwpp PRIVATE ${FFTW_LIBRARIES})
target_link_libraries(fftwpp PRIVATE MPI::MPI_CXX)
target_include_directories(fftwpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
