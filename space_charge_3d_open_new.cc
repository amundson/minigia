#include "space_charge_3d_open_new.h"
#include "core_diagnostics.h"

Space_charge_3d_open_new::Space_charge_3d_open_new(
        std::vector<int> const& grid_shape, double n_sigma):
    grid_shape(grid_shape)
  , n_sigma(n_sigma)
  , domain(grid_shape, false)
  , domain_fixed(false)
{

}

namespace {
double
get_smallest_non_tiny(double val, double other1, double other2, double tiny)
{
    double retval;
    if (val > tiny) {
        retval = val;
    } else {
        if ((other1 > tiny) && (other2 > tiny)) {
            retval = std::min(other1, other2);
        } else {
            retval = std::max(other1, other2);
        }
    }
    return retval;
}
}

void
Space_charge_3d_open_new::update_domain(Bunch const& bunch)
{
    if (!domain_fixed) {
        MArray1d mean(Core_diagnostics::calculate_mean(bunch));
        MArray1d std(Core_diagnostics::calculate_std(bunch, mean));
        std::vector<double> size(3);
        std::vector<double> offset(3);
        const double tiny = 1.0e-10;
        if ((std[Bunch::x] < tiny) && (std[Bunch::y] < tiny) &&
            (std[Bunch::z] < tiny)) {
            throw std::runtime_error("Space_charge_3d_open_hockney::update_"
                                     "domain: all three spatial dimensions "
                                     "have neglible extent");
        }
        offset[0] = mean[Bunch::z];
        size[0] = n_sigma * get_smallest_non_tiny(std[Bunch::z], std[Bunch::x],
                                                  std[Bunch::y], tiny);
        offset[1] = mean[Bunch::y];
        size[1] = n_sigma * get_smallest_non_tiny(std[Bunch::y], std[Bunch::x],
                                                  std[Bunch::z], tiny);
        offset[2] = mean[Bunch::x];
        size[2] = n_sigma * get_smallest_non_tiny(std[Bunch::x], std[Bunch::y],
                                                  std[Bunch::z], tiny);
        domain.set_physical_offset(offset);
        domain.set_physical_size(size);
    }
}

Space_charge_3d_open_new::~Space_charge_3d_open_new()
{

}
