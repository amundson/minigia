#ifndef RECTANGULAR_GRID_DOMAIN_EIGEN_H_
#define RECTANGULAR_GRID_DOMAIN_EIGEN_H_

#include "fast_int_floor.h"
#include <array>
#include <boost/shared_ptr.hpp>

class Rectangular_grid_domain_eigen
{
private:
    std::array<double, 3> physical_size;
    std::array<double, 3> physical_offset;
    std::array<int, 3> grid_shape;
    bool periodic_z;

    std::array<double, 3> left;
    std::array<double, 3> cell_size;

    void init_left_cell_size();

public:
    Rectangular_grid_domain_eigen(std::array<double, 3> const& physical_size,
                                  std::array<double, 3> const& physical_offset,
                                  std::array<int, 3> const& grid_shape,
                                  bool periodic_z);
    Rectangular_grid_domain_eigen(std::array<double, 3> const& physical_size,
                                  std::array<double, 3> const& physical_offset,
                                  std::array<int, 3> const& grid_shape);

    Rectangular_grid_domain_eigen(std::array<double, 3> const& physical_size,
                                  std::array<int, 3> const& grid_shape,
                                  bool periodic_z);

    std::array<double, 3> const& get_physical_size() const;
    std::array<double, 3> const& get_physical_offset() const;
    std::array<int, 3> const& get_grid_shape() const;
    std::array<double, 3> const& get_cell_size() const;
    bool is_periodic() const;
    std::array<double, 3> const& get_left() const;
    // returns cell location and fractional offset
    inline bool get_leftmost_indices_offsets(double x, double y, double z,
                                             int& ix, int& iy, int& iz,
                                             double& offx, double& offy,
                                             double& offz) const
    {
        bool retval;
        double scaled_location;

        scaled_location = (x - left[0]) / cell_size[0] - 0.5;
        ix = fast_int_floor(scaled_location);
        offx = scaled_location - ix;

        scaled_location = (y - left[1]) / cell_size[1] - 0.5;
        iy = fast_int_floor(scaled_location);
        offy = scaled_location - iy;

        scaled_location = (z - left[2]) / cell_size[2] - 0.5;
        iz = fast_int_floor(scaled_location);
        offz = scaled_location - iz;

        if (grid_shape[2] == 1) {
            // csp: For grid_shape = 1, iz and offz are not used in deposit
            //      and interpolation. These are just for the reference.
            //      iz is 0 or 1, so that all particles are in domain, i.e.,
            //      no cutting edge.
            iz += 1;
            if (iz == 0)
                offz = -0.5 + offz;
            if (iz == 1)
                offz = 0.5 - offz;
            retval = ((ix >= 0) && (ix < grid_shape[0] - 1) && (iy >= 0) &&
                      (iy < grid_shape[1] - 1)) &&
                     (periodic_z || ((iz >= 0) && (iz <= grid_shape[2])));
        } else {
            retval = ((ix >= 0) && (ix < grid_shape[0] - 1) && (iy >= 0) &&
                      (iy < grid_shape[1] - 1)) &&
                     (periodic_z || ((iz >= 0) && (iz < grid_shape[2] - 1)));
        }

        return retval;
    }

    void get_cell_coordinates(int ix, int iy, int iz, double& x, double& y,
                              double& z) const;
};

typedef boost::shared_ptr<Rectangular_grid_domain_eigen>
    Rectangular_grid_domain_eigen_sptr; // syndoc:include

#endif /* RECTANGULAR_GRID_DOMAIN_EIGEN_H_ */
