cmake_minimum_required(VERSION 3.9)
project(MINIGIA)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH "${MINIGIA_SOURCE_DIR}/CMake")

if(DEFINES)
  add_definitions(${DEFINES})
endif()

find_package(MPI REQUIRED CXX)
find_package(Boost 1.41 REQUIRED)
find_package(OpenMP)

option(SC "compile space charge benchmarks; requires FFTW" OFF)
#if(SC)
find_package(FFTW REQUIRED MPI)
#endif()

# chef
option(CHEF "Use chef libraries" OFF)
if(CHEF)
find_package(CHEF REQUIRED)
INCLUDE_DIRECTORIES(${CHEF_INCLUDE_DIR})
LINK_DIRECTORIES(${CHEF_LIB_DIR})
add_definitions(-DHAVE_CHEF)
endif()

configure_file("${MINIGIA_SOURCE_DIR}/bunch_data_paths.h.in"
               "${MINIGIA_BINARY_DIR}/bunch_data_paths.h" IMMEDIATE)

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
target_include_directories(independent_particle PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(independent_particle PRIVATE vectorclass)
target_include_directories(independent_particle PRIVATE eigen)
target_link_libraries(independent_particle PRIVATE MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(independent_particle PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(independent_particle PRIVATE ${CHEF_LIBS})
target_link_libraries(independent_particle PRIVATE boost_system)

add_executable(populate
    populate.cc
    bunch.h
    commxx.h
    commxx.cc)
target_include_directories(populate PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(populate PRIVATE eigen)
target_link_libraries(populate PRIVATE MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(populate PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(fftwpp)

add_executable(fft
    fft-main.cc
    commxx.cc
    commxx.h
    distributed_fft3d.h
    distributed_fft3d.cc
    multi_array_typedefs.h)
target_link_libraries(fft PRIVATE ${FFTW_LIBRARIES})
target_link_libraries(fft PRIVATE MPI::MPI_CXX)
target_include_directories(fft PRIVATE eigen)
target_link_libraries(fft PRIVATE fftwpp)

if(SC)
add_executable(space_charge_3d
               space_charge_3d.cc
               space_charge_3d_open_hockney.cc
               space_charge_3d_open_hockney.h
               space_charge_3d_open_hockney_eigen.cc
               space_charge_3d_open_hockney_eigen.h
               commxx.cc
               commxx.h
               commxx_divider.cc
               commxx_divider.h
               core_diagnostics.cc
               core_diagnostics.h
               deposit.cc
               deposit.h
               distributed_fft3d.cc
               distributed_fft3d.h
               distributed_rectangular_grid.cc
               distributed_rectangular_grid.h
               distributed_rectangular_grid_eigen.cc
               distributed_rectangular_grid_eigen.h
               interpolate_rectangular_zyx.h
               math_constants.h
               multi_array_offsets.h
               rectangular_grid.cc
               rectangular_grid.h
               rectangular_grid_eigen.h
               rectangular_grid_domain.cc
               rectangular_grid_domain.h
               rectangular_grid_domain_eigen.cc
               rectangular_grid_domain_eigen.h
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
target_include_directories(space_charge_3d PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(space_charge_3d PRIVATE vectorclass)
target_include_directories(space_charge_3d PRIVATE eigen)
target_link_libraries(space_charge_3d PRIVATE MPI::MPI_CXX)
target_link_libraries(space_charge_3d PRIVATE ${FFTW_LIBRARIES})
endif()

add_subdirectory(tests)
