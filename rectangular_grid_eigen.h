#ifndef RECTANGULAR_GRID_EIGEN_H_
#define RECTANGULAR_GRID_EIGEN_H_
#include "multi_array_typedefs.h"
#include "rectangular_grid_domain_eigen.h"
#include <unsupported/Eigen/CXX11/Tensor>

class Rectangular_grid_eigen
{
public:
    typedef Eigen::Tensor<double, 3> Grid_points_t;

private:
    Rectangular_grid_domain_eigen domain;
    Grid_points_t grid_points;
    double normalization;

public:
    Rectangular_grid_eigen(std::array<double, 3> const& physical_size,
                           std::array<double, 3> const& physical_offset,
                           std::array<int, 3> const& grid_shape,
                           bool periodic_z)
        : domain(physical_size, physical_offset, grid_shape, periodic_z)
        , grid_points(grid_shape[0], grid_shape[1], grid_shape[2])
        , normalization(1.0)
    {}

    Rectangular_grid_eigen(
        Rectangular_grid_domain_eigen const& rectangular_grid_domain_eigen)
        : domain(rectangular_grid_domain_eigen)
        , grid_points(domain.get_grid_shape()[0], domain.get_grid_shape()[1],
                      domain.get_grid_shape()[2])
        , normalization(1.0)
    {}

    Rectangular_grid_domain_eigen const& get_domain() const { return domain; }

    Rectangular_grid_domain_eigen& get_domain() { return domain; }

    Grid_points_t const& get_grid_points() const { return grid_points; }

    Grid_points_t& get_grid_points() { return grid_points; }

    void set_normalization(double val) { normalization = val; }

    double get_normalization() const { return normalization; }

    double get_interpolated(std::array<double, 3> location) const
    {
        return get_interpolated_coord(location[0], location[1], location[2]);
    }

    double get_interpolated_coord(double x, double y, double z) const
    {
        // tri-linear interpolation
        int ix, iy, iz;
        double offx, offy, offz;
        domain.get_leftmost_indices_offsets(x, y, z, ix, iy, iz, offx, offy,
                                            offz);
        Grid_points_t const& a(grid_points);
        double val = 0.0;
        if ((domain.get_grid_shape()[0] > 1) &&
            (domain.get_grid_shape()[1] > 1) &&
            (domain.get_grid_shape()[2] > 1)) {
            if ((ix < 0) || (ix >= domain.get_grid_shape()[0] - 1) ||
                (iy < 0) || (iy >= domain.get_grid_shape()[1] - 1) ||
                (iz < 0) || (iz >= domain.get_grid_shape()[2] - 1)) {
                val = 0.0;
            } else {
                val = ((1.0 - offx) * (1.0 - offy) * (1.0 - offz) *
                           a(ix, iy, iz) +
                       (1.0 - offx) * (1.0 - offy) * offz * a(ix, iy, iz + 1) +
                       (1.0 - offx) * offy * (1.0 - offz) * a(ix, iy + 1, iz) +
                       (1.0 - offx) * offy * offz * a(ix, iy + 1, iz + 1) +
                       offx * (1.0 - offy) * (1.0 - offz) * a(ix + 1, iy, iz) +
                       offx * (1.0 - offy) * offz * a(ix + 1, iy, iz + 1) +
                       offx * offy * (1.0 - offz) * a(ix + 1, iy + 1, iz) +
                       offx * offy * offz * a(ix + 1, iy + 1, iz + 1));
            }
        } else if (domain.get_grid_shape()[0] == 1) {
            // 2D,  Y-Z plane
            if ((iy < 0) || (iy >= domain.get_grid_shape()[1] - 1) ||
                (iz < 0) || (iz >= domain.get_grid_shape()[2] - 1)) {
                val = 0.0;
            } else {
                val = ((1.0 - offz) * (1.0 - offy) * a(ix, iy, iz) +
                       offy * (1.0 - offz) * a(ix, iy + 1, iz) +
                       (1.0 - offy) * offz * a(ix, iy, iz + 1) +
                       offy * offz * a(ix, iy + 1, iz + 1));
            }
        } else if (domain.get_grid_shape()[1] == 1) {
            // 2D,  X-Z plane
            if ((ix < 0) || (ix >= domain.get_grid_shape()[0] - 1) ||
                (iz < 0) || (iz >= domain.get_grid_shape()[2] - 1)) {
                val = 0.0;
            } else {
                val = ((1.0 - offz) * (1.0 - offx) * a(ix, iy, iz) +
                       offx * (1.0 - offz) * a(ix + 1, iy, iz) +
                       (1.0 - offx) * offz * a(ix, iy, iz + 1) +
                       offx * offz * a(ix + 1, iy, iz + 1));
            }

        } else if (domain.get_grid_shape()[2] == 1) {
            // 2D,  X-Y plane
            if ((ix < 0) || (ix >= domain.get_grid_shape()[0] - 1) ||
                (iy < 0) || (iy >= domain.get_grid_shape()[1] - 1)) {
                val = 0.0;
            } else {
                val = ((1.0 - offy) * (1.0 - offx) * a(ix, iy, iz) +
                       offx * (1.0 - offy) * a(ix + 1, iy, iz) +
                       (1.0 - offx) * offy * a(ix, iy + 1, iz) +
                       offx * offy * a(ix + 1, iy + 1, iz));
            }
        }
        return val;
    }
};

typedef boost::shared_ptr<Rectangular_grid_eigen>
    Rectangular_grid_eigen_sptr; // syndoc:include

#endif /* RECTANGULAR_GRID_EIGEN_H_ */
