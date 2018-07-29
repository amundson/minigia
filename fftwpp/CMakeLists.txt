add_library(fftwpp
    fftw++.cc)
target_link_libraries(fftwpp PRIVATE ${FFTW_LIBRARIES})
target_link_libraries(fftwpp PRIVATE MPI::MPI_CXX)
target_include_directories(fftwpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
