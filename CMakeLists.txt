project(MINIGIA)
cmake_minimum_required(VERSION 3.2)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})

add_executable(independent_particle
    independent_particle.cc
    bunch.h
    four_momentum.h
    multi_array_typedefs.h
    reference_particle.h
    restrict_extension.h)
