#ifndef RECTANGULAR_GRID_DOMAIN_FIXTURE_EIGEN_H_
#define RECTANGULAR_GRID_DOMAIN_FIXTURE_EIGEN_H_

#include "rectangular_grid_domain_eigen.h"

const double domain_min = -1.0;
const double domain_max = 1.0;
const double domain_size = domain_max - domain_min;
const double domain_offset = 5.0;
const int grid_size0 = 4;
const int grid_size1 = 5;
const int grid_size2 = 3;
const bool is_periodic = false;

struct Rectangular_grid_domain_eigen_fixture
{
    Rectangular_grid_domain_eigen_fixture() :
        physical_size{domain_size, domain_size, domain_size},
        physical_offset{domain_offset, domain_offset, domain_offset},
        grid_shape{grid_size0, grid_size1, grid_size2},
        rectangular_grid_domain_eigen(physical_size, physical_offset,
                                      grid_shape, is_periodic)
    {
    }

    ~Rectangular_grid_domain_eigen_fixture()
    {
    }

    std::array<double, 3> physical_size, physical_offset;
    std::array<int, 3> grid_shape;
    Rectangular_grid_domain_eigen rectangular_grid_domain_eigen;
};

#endif /* RECTANGULAR_GRID_DOMAIN_FIXTURE_EIGEN_H_ */
