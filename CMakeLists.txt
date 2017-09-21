project(MINIGIA)
cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH "${MINIGIA_SOURCE_DIR}/CMake")

set (CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if(DEFINES)
    add_definitions(${DEFINES})
endif()

find_package(MPI REQUIRED)
find_package(Boost 1.58 REQUIRED)
find_package (Eigen3 3.2 REQUIRED NO_MODULE)

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
target_link_libraries(independent_particle PRIVATE MPI::MPI_CXX)
target_link_libraries(independent_particle PRIVATE Eigen3::Eigen)

