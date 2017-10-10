project(MINIGIA)
cmake_minimum_required(VERSION 3.9)
list(APPEND CMAKE_MODULE_PATH "${MINIGIA_SOURCE_DIR}/CMake")

set (CMAKE_CXX_STANDARD 11)

if(DEFINES)
    add_definitions(${DEFINES})
endif()

find_package(MPI REQUIRED CXX)
find_package(Boost 1.53 REQUIRED)
find_package(OpenMP) 

if(FALSE)
find_package(FFTW3 REQUIRED)
endif()

add_executable(independent_particle
    independent_particle.cc
    bunch.h
    commxx.cc
    commxx.h
    four_momentum.h
    gsvector.h
    multi_array_typedefs.h
    reference_particle.h
    restrict_extension.h
    )

target_include_directories(independent_particle PRIVATE vectorclass)
target_include_directories(independent_particle PRIVATE eigen)
target_link_libraries(independent_particle PRIVATE MPI::MPI_CXX)
target_link_libraries(independent_particle PRIVATE OpenMP::OpenMP_CXX)

