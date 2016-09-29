project(MINIGIA)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH "${MINIGIA_SOURCE_DIR}/CMake")

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

if(DEFINES)
    add_definitions(${DEFINES})
endif()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})
set(PARALLEL_FFTW_LIBRARIES ${FFTW3_MPI_LIBRARIES} ${FFTW3_OMP_LIBRARIES} ${FFTW3_LIBRARIES})

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/vectorclass)

add_executable(independent_particle
    independent_particle.cc
    bunch.h
    commxx.cc
    commxx.h
    four_momentum.h
    gsvector.h
    multi_array_typedefs.h
    reference_particle.h
    restrict_extension.h)
target_link_libraries(independent_particle ${MPI_LIBRARIES})

add_executable(space_charge_3d
    space_charge_3d.cc
    space_charge_3d_open_hockney.cc
    space_charge_3d_open_hockney.h
    space_charge_3d_open_new.cc
    space_charge_3d_open_new.h
    Array.h
    array_typedefs.h
    commxx.cc
    commxx.h
    commxx_divider.cc
    commxx_divider.h
    core_diagnostics.cc
    core_diagnostics.h
    deposit.cc
    deposit.h
    deposit_new.cc
    deposit_new.h
    distributed_fft3d.cc
    distributed_fft3d.h
    distributed_rectangular_grid.cc
    distributed_rectangular_grid.h
    interpolate_rectangular_zyx.h
    math_constants.h
    multi_array_offsets.h
    rectangular_grid.cc
    rectangular_grid.h
    rectangular_grid_domain.cc
    rectangular_grid_domain.h
    simple_timer.h
    space_charge_3d.cc
    space_charge_3d_open_hockney.cc
    bunch.h
    four_momentum.h
    gsvector.h
    multi_array_typedefs.h
    physical_constants.h
    reference_particle.h
    restrict_extension.h)
target_link_libraries(space_charge_3d ${MPI_LIBRARIES}
    ${PARALLEL_FFTW_LIBRARIES})

