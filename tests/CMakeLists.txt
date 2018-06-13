if(SC)
add_executable(test_distributed_rectangular_grid_eigen
    test_distributed_rectangular_grid_eigen.cc
    ../distributed_rectangular_grid_eigen.cc
    ../rectangular_grid_domain_eigen.cc
    ../commxx.cc
    )

#target_include_directories(test_distributed_rectangular_grid_eigen PRIVATE ${MINIGIA_BINARY_DIR})
target_include_directories(test_distributed_rectangular_grid_eigen PRIVATE ${MINIGIA_SOURCE_DIR}/eigen)
target_include_directories(test_distributed_rectangular_grid_eigen PRIVATE ${MINIGIA_SOURCE_DIR})
target_include_directories(test_distributed_rectangular_grid_eigen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_distributed_rectangular_grid_eigen PRIVATE MPI::MPI_CXX)
endif()

