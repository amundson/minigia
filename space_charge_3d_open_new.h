#ifndef SPACE_CHARGE_3D_OPEN_NEW_H
#define SPACE_CHARGE_3D_OPEN_NEW_H

#include <vector>

#include "bunch.h"
#include "rectangular_grid_domain.h"

class Space_charge_3d_open_new
{
private:
    std::vector<int> grid_shape;
    double n_sigma;
    Rectangular_grid_domain domain;
    bool domain_fixed;

public:
    Space_charge_3d_open_new(std::vector<int> const& grid_shape,
                             double n_sigma = 8.0);
    void update_domain(Bunch const& bunch);
    Rectangular_grid_domain const& get_domain() { return domain; }
    void apply(Bunch& bunch, double time_step, int verbosity);
    virtual ~Space_charge_3d_open_new();
};

#endif // SPACE_CHARGE_3D_OPEN_NEW_H
