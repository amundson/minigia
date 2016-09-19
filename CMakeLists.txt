project(MINIGIA)
cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(independent_particle
    independent_particle.cc
    bunch.h
    four_momentum.h
    reference_particle.h
    restrict_extension.h)
