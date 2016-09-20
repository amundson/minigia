project(MINIGIA)
cmake_minimum_required(VERSION 3.2)

find_package(MPI REQUIRED)

if(DEFINES)
    add_definitions(${DEFINES})
endif()

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/vectorclass)

add_executable(independent_particle
    independent_particle.cc
    bunch.h
    four_momentum.h
    gsvector.h
    multi_array_typedefs.h
    reference_particle.h
    restrict_extension.h)
target_link_libraries(independent_particle ${MPI_LIBRARIES})

