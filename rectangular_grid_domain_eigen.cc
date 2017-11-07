#include "rectangular_grid_domain_eigen.h"

void
Rectangular_grid_domain_eigen::init_left_cell_size()
{
    for (size_t i = 0; i < 3; ++i) {
        left[i] = physical_offset[i] - physical_size[i] / 2.0;
        cell_size[i] = physical_size[i] / (1.0 * grid_shape[i]);
    }
}

Rectangular_grid_domain_eigen::Rectangular_grid_domain_eigen(
    std::array<double, 3> const& physical_size,
    std::array<double, 3> const& physical_offset,
    std::array<int, 3> const& grid_shape, bool periodic_z)
    : physical_size(physical_size)
    , physical_offset(physical_offset)
    , grid_shape(grid_shape)
    , periodic_z(periodic_z)
{}

Rectangular_grid_domain_eigen::Rectangular_grid_domain_eigen(
    std::array<double, 3> const& physical_size,
    std::array<double, 3> const& physical_offset,
    std::array<int, 3> const& grid_shape)
    : physical_size(physical_size)
    , physical_offset(physical_offset)
    , grid_shape(grid_shape)
    , periodic_z(false)
{}

Rectangular_grid_domain_eigen::Rectangular_grid_domain_eigen(
    std::array<double, 3> const& physical_size,
    std::array<int, 3> const& grid_shape, bool periodic_z)
    : physical_size(physical_size)
    , physical_offset{ { 0, 0, 0 } }
    , grid_shape(grid_shape)
    , periodic_z(periodic_z)
{}

std::array<double, 3> const&
Rectangular_grid_domain_eigen::get_physical_size() const
{
    return physical_size;
}

std::array<double, 3> const&
Rectangular_grid_domain_eigen::get_physical_offset() const
{
    return physical_offset;
}

std::array<int, 3> const&
Rectangular_grid_domain_eigen::get_grid_shape() const
{
    return grid_shape;
}

std::array<double, 3> const&
Rectangular_grid_domain_eigen::get_cell_size() const
{
    return cell_size;
}

bool
Rectangular_grid_domain_eigen::is_periodic() const
{
    return periodic_z;
}
std::array<double, 3> const&
Rectangular_grid_domain_eigen::get_left() const
{
    return left;
}

void
Rectangular_grid_domain_eigen::get_cell_coordinates(int ix, int iy, int iz,
                                                    double& x, double& y,
                                                    double& z) const
{
    x = left[0] + cell_size[0] * (0.5 + ix);
    y = left[1] + cell_size[1] * (0.5 + iy);
    z = left[2] + cell_size[2] * (0.5 + iz);
}
