#include "space_charge_3d_open_hockney_eigen.h"

#include <stdexcept>
#include <cstring>

#include "core_diagnostics.h"
#include "math_constants.h"
using mconstants::pi;
#include "physical_constants.h"
using pconstants::epsilon0;
#include "deposit.h"
#include "interpolate_rectangular_zyx.h"
#include "multi_array_offsets.h"
#include "simple_timer.h"

void
Space_charge_3d_open_hockney_eigen::setup_communication(
        Commxx_sptr const& bunch_comm_sptr)
{
    if (comm2_sptr != commxx_divider_sptr->get_commxx_sptr(bunch_comm_sptr)) {
        comm2_sptr = commxx_divider_sptr->get_commxx_sptr(bunch_comm_sptr);
        setup_derived_communication();
    }
}

void
Space_charge_3d_open_hockney_eigen::setup_derived_communication()
{
    distributed_fft3d_sptr = Distributed_fft3d_sptr(
            new Distributed_fft3d(doubled_grid_shape, comm2_sptr));
    padded_grid_shape = distributed_fft3d_sptr->get_padded_shape_real();
    std::vector<int > ranks1; // ranks with data from the undoubled domain
    int lower = 0;
    for (int rank = 0; rank < comm2_sptr->get_size(); ++rank) {
        int uppers2 = distributed_fft3d_sptr->get_uppers()[rank];
        int uppers1 = std::min(uppers2, grid_shape[0]);
        int length0;
        if (rank > 0) {
            length0 = uppers1 - distributed_fft3d_sptr->get_uppers()[rank - 1];
        } else {
            length0 = uppers1;
        }
        if (length0 > 0) {
            ranks1.push_back(rank);
            lowers1.push_back(lower);
            int total_length = length0 * grid_shape[1] * grid_shape[2];
            lengths1.push_back(total_length);
            lower += total_length;
        }
    }
    comm1_sptr = Commxx_sptr(new Commxx(comm2_sptr, ranks1));
    std::vector<int > real_uppers(distributed_fft3d_sptr->get_uppers());
    real_lengths = distributed_fft3d_sptr->get_lengths();
    for (int i = 0; i < comm2_sptr->get_size(); ++i) {
        if (real_uppers[i] > grid_shape[0]) {
            real_uppers[i] = grid_shape[0];
        }
        if (i == 0) {
            real_lengths[0] = real_uppers[0] * grid_shape[1] * grid_shape[2];
        } else {
            real_lengths[i] = (real_uppers[i] - real_uppers[i - 1])
                    * grid_shape[1] * grid_shape[2];
        }
    }
    int my_rank = comm2_sptr->get_rank();
    if (my_rank > 0) {
        real_lower = real_uppers[my_rank - 1];
    } else {
        real_lower = 0;
    }
    real_upper = real_uppers[my_rank];
    real_length = real_lengths[my_rank];
    if (my_rank > 0) {
        doubled_lower = distributed_fft3d_sptr->get_uppers()[my_rank - 1];
    } else {
        doubled_lower = 0;
    }
    doubled_upper = distributed_fft3d_sptr->get_uppers()[my_rank];
    real_doubled_lower = std::min(doubled_lower, grid_shape[0]);
    real_doubled_upper = std::min(doubled_upper, grid_shape[0]);
}

void
Space_charge_3d_open_hockney_eigen::constructor_common(
        std::vector<int > const& grid_shape)
{
    this->grid_shape[0] = grid_shape[2];
    this->grid_shape[1] = grid_shape[1];
    this->grid_shape[2] = grid_shape[0];
    for (int i = 0; i < 3; ++i) {
        doubled_grid_shape[i] = 2 * this->grid_shape[i];
    }
}

Space_charge_3d_open_hockney_eigen::Space_charge_3d_open_hockney_eigen(
        Commxx_divider_sptr commxx_divider_sptr,
        std::vector<int > const & grid_shape,
        double n_sigma) :
                grid_shape(3),
                doubled_grid_shape(3),
                padded_grid_shape(3),
                commxx_divider_sptr(commxx_divider_sptr),
                comm2_sptr(),
                comm1_sptr(),
                n_sigma(n_sigma),
                domain_fixed(false),
                have_domains(false)
{
    constructor_common(grid_shape);
}

double
Space_charge_3d_open_hockney_eigen::get_n_sigma() const
{
    return n_sigma;
}

void
Space_charge_3d_open_hockney_eigen::set_doubled_domain()
{
    std::vector<double > doubled_size(3);
    for (int i = 0; i < 3; ++i) {
        doubled_size[i] = 2 * domain_sptr->get_physical_size()[i];
    }
    doubled_domain_sptr = Rectangular_grid_domain_sptr(
            new Rectangular_grid_domain(doubled_size,
                    domain_sptr->get_physical_offset(), doubled_grid_shape,
                    false));
}

namespace{
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
Space_charge_3d_open_hockney_eigen::update_domain(Bunch const& bunch)
{
    setup_communication(bunch.get_comm_sptr());
    if (!domain_fixed) {
        MArray1d mean(Core_diagnostics::calculate_mean(bunch));
        MArray1d std(Core_diagnostics::calculate_std(bunch, mean));
        std::vector<double > size(3);
        std::vector<double > offset(3);
        const double tiny = 1.0e-10;
        if ((std[Bunch::x] < tiny) && (std[Bunch::y] < tiny)
                && (std[Bunch::z] < tiny)) {
            throw std::runtime_error(
                    "Space_charge_3d_open_hockney_eigen::update_domain: all three spatial dimensions have neglible extent");
        }
        offset[0] = mean[Bunch::z];
        size[0] = n_sigma
                * get_smallest_non_tiny(std[Bunch::z], std[Bunch::x],
                        std[Bunch::y], tiny);
        offset[1] = mean[Bunch::y];
        size[1] = n_sigma
                * get_smallest_non_tiny(std[Bunch::y], std[Bunch::x],
                        std[Bunch::z], tiny);
        offset[2] = mean[Bunch::x];
        size[2] = n_sigma
                * get_smallest_non_tiny(std[Bunch::x], std[Bunch::y],
                        std[Bunch::z], tiny);
        domain_sptr = Rectangular_grid_domain_sptr(
                new Rectangular_grid_domain(size, offset, grid_shape,
                        false));
        set_doubled_domain();
        have_domains = true;
    }
}

Rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_local_charge_density(Bunch const& bunch)
{
double t = simple_timer_current();
    update_domain(bunch);
t = simple_timer_show(t, "sc-local-rho-update-domain");
    Rectangular_grid_sptr local_rho_sptr(new Rectangular_grid(domain_sptr));
t = simple_timer_show(t, "sc-local-rho-new");
    deposit_charge_rectangular_zyx(*local_rho_sptr, bunch);
    //deposit_charge_rectangular_zyx_omp_reduce(*local_rho_sptr, bunch);
    //deposit_charge_rectangular_zyx_omp_interleaved(*local_rho_sptr, bunch);
t = simple_timer_show(t, "sc-local-rho-deposit");
    return local_rho_sptr;
}


Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_global_charge_density2_allreduce(
        Rectangular_grid const& local_charge_density, Commxx_sptr comm_sptr)
{
    setup_communication(comm_sptr);
    int error = MPI_Allreduce(MPI_IN_PLACE,
            (void*) local_charge_density.get_grid_points().origin(),
            local_charge_density.get_grid_points().num_elements(), MPI_DOUBLE,
            MPI_SUM, comm_sptr->get());
    if (error != MPI_SUCCESS) {
        throw std::runtime_error(
                "MPI error in Space_charge_3d_open_hockney_eigen::get_global_charge_density2_allreduce");
    }
    Distributed_rectangular_grid_sptr rho2 = Distributed_rectangular_grid_sptr(
            new Distributed_rectangular_grid(doubled_domain_sptr, doubled_lower,
                    doubled_upper,
                    distributed_fft3d_sptr->get_padded_shape_real(),
                    comm_sptr));
    for (int i = rho2->get_lower(); i < rho2->get_upper(); ++i) {
        for (int j = 0; j < doubled_grid_shape[1]; ++j) {
            for (int k = 0; k < doubled_grid_shape[2]; ++k) {
                rho2->get_grid_points()[i][j][k] = 0.0;
            }
        }
    }
    for (int i = real_lower; i < real_upper; ++i) {
        for (int j = 0; j < grid_shape[1]; ++j) {
            for (int k = 0; k < grid_shape[2]; ++k) {
                rho2->get_grid_points()[i][j][k] =
                        local_charge_density.get_grid_points()[i][j][k];
            }
        }
    }
    return rho2;
}

Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_global_charge_density2(
        Rectangular_grid const& local_charge_density, Commxx_sptr comm_sptr)
{
    return get_global_charge_density2_allreduce(local_charge_density,
            comm_sptr);
}

Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_green_fn2_pointlike()
{
    if (doubled_domain_sptr == NULL) {
        throw std::runtime_error(
                "Space_charge_3d_open_hockney_eigen::get_green_fn2_pointlike called before domain specified");
    }
    int lower = distributed_fft3d_sptr->get_lower();
    int upper = distributed_fft3d_sptr->get_upper();
    Distributed_rectangular_grid_sptr G2 = Distributed_rectangular_grid_sptr(
            new Distributed_rectangular_grid(doubled_domain_sptr, lower, upper,
                    distributed_fft3d_sptr->get_padded_shape_real(),
                    comm2_sptr));

    double hx = domain_sptr->get_cell_size()[2];
    double hy = domain_sptr->get_cell_size()[1];
    double hz = domain_sptr->get_cell_size()[0];

// G000 is naively infinite. In the correct approach, it should be
// the value which gives the proper integral when convolved with the
// charge density. Even assuming a constant charge density, the proper
// value for G000 cannot be computed in closed form. Fortunately,
// the solver results are insensitive to the exact value of G000.
// I make the following argument: G000 should be greater than any of
// the neighboring values of G. The form
//    G000 = coeff/min(hx,hy,hz),
// with
//    coeff > 1
// satisfies the criterion. An empirical study (see the 3d_open_hockney_eigen.py
// script in docs/devel/solvers) gives coeff = 2.8.
    const double coeff = 2.8;
    double G000 = coeff / std::min(hx, std::min(hy, hz));

    const int num_images = 8;
    int mix, miy; // mirror indices for x- and y-planes
    double dx, dy, dz, G;

// In the following loops we use mirroring for ix and iy, but
// calculate all iz values separately because the mirror points in
// iz may be on another processor.
// Note that the doubling algorithm is not quite symmetric. For
// example, the doubled grid for 4 points in 1d looks like
//    0 1 2 3 4 3 2 1

    #pragma omp parallel for private( dx, dy, dz, G, mix, miy )
    for (int iz = lower; iz < upper; ++iz) {
        if (iz > grid_shape[0]) {
            dz = (doubled_grid_shape[0] - iz) * hz;
        } else {
            dz = iz * hz;
        }
        for (int iy = 0; iy < grid_shape[1] + 1; ++iy) {
            dy = iy * hy;
            miy = doubled_grid_shape[1] - iy;
            if (miy == doubled_grid_shape[1]) {
                miy = iy;
            }
            for (int ix = 0; ix < grid_shape[2] + 1; ++ix) {
                dx = ix * hx;
                mix = doubled_grid_shape[2] - ix;
                if (mix == doubled_grid_shape[2]) {
                    mix = ix;
                }
                if ((ix == 0) && (iy == 0) && (iz == 0)) {
                    G = G000;
                } else {
                    G = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);
                }
                G2->get_grid_points()[iz][iy][ix] = G;
                // three mirror images
                G2->get_grid_points()[iz][miy][ix] = G;
                G2->get_grid_points()[iz][miy][mix] = G;
                G2->get_grid_points()[iz][iy][mix] = G;
            }
        }
    }

    G2->set_normalization(1.0);

    return G2;
}

Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_scalar_field2(
        Distributed_rectangular_grid & charge_density2,
        Distributed_rectangular_grid & green_fn2)
{
    std::vector<int > cshape(
            distributed_fft3d_sptr->get_padded_shape_complex());
    int lower = distributed_fft3d_sptr->get_lower();
    int upper = distributed_fft3d_sptr->get_upper();

    MArray3dc rho2hat(
            boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    MArray3dc G2hat(
            boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    MArray3dc phi2hat(
            boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    distributed_fft3d_sptr->transform(charge_density2.get_grid_points(),
            rho2hat);
    distributed_fft3d_sptr->transform(green_fn2.get_grid_points(), G2hat);

    #pragma omp parallel for
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < cshape[1]; ++j) {
            for (int k = 0; k < cshape[2]; ++k) {
                phi2hat[i][j][k] = rho2hat[i][j][k] * G2hat[i][j][k];
            }
        }
    }

    double hx, hy, hz;
    hx = domain_sptr->get_cell_size()[2];
    hy = domain_sptr->get_cell_size()[1];
    hz = domain_sptr->get_cell_size()[0];
    double normalization = hx * hy * hz; // volume element in integral
    normalization *= 1.0 / (4.0 * pi * epsilon0);

    Distributed_rectangular_grid_sptr phi2(
            new Distributed_rectangular_grid(doubled_domain_sptr, lower, upper,
                    distributed_fft3d_sptr->get_padded_shape_real(), comm2_sptr));

    distributed_fft3d_sptr->inv_transform(phi2hat, phi2->get_grid_points());

    normalization *= charge_density2.get_normalization();
    normalization *= green_fn2.get_normalization();
    normalization *= distributed_fft3d_sptr->get_roundtrip_normalization();
    phi2->set_normalization(normalization);

    return phi2;
}

Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::extract_scalar_field(
        Distributed_rectangular_grid const & phi2)
{
    Distributed_rectangular_grid_sptr phi(
            new Distributed_rectangular_grid(domain_sptr, real_doubled_lower,
                    real_doubled_upper, comm1_sptr));

    #pragma omp parallel for
    for (int i = real_doubled_lower; i < real_doubled_upper; ++i) {
        for (int j = 0; j < grid_shape[1]; ++j) {
            for (int k = 0; k < grid_shape[2]; ++k) {
                phi->get_grid_points()[i][j][k] =
                        phi2.get_grid_points()[i][j][k];
            }
        }
    }
    phi->set_normalization(phi2.get_normalization());
    if (comm1_sptr->has_this_rank()) {
        phi->fill_guards();
    }
    return phi;
}

Distributed_rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_electric_field_component(
        Distributed_rectangular_grid const& phi, int component)
{
    int index;
    if (component == 0) {
        index = 2;
    } else if (component == 1) {
        index = 1;
    } else if (component == 2) {
        index = 0;
    } else {
        throw std::runtime_error(
                "Space_charge_3d_open_hockney_eigen::get_electric_field_component: component must be 0, 1 or 2");
    }

    Distributed_rectangular_grid_sptr En(
            new Distributed_rectangular_grid(domain_sptr, phi.get_lower(),
                    phi.get_upper(), comm1_sptr));
    MArray3d_ref En_a(En->get_grid_points());
    MArray3d_ref phi_a(phi.get_grid_points());
    int lower_limit, upper_limit;
    if (index == 0) {
        lower_limit = En->get_lower_guard();
        upper_limit = En->get_upper_guard();
    } else {
        lower_limit = 0;
        upper_limit = domain_sptr->get_grid_shape()[index];
    }
    double cell_size = domain_sptr->get_cell_size()[index];
    boost::array<MArray3d::index, 3 > center, left, right;

    #pragma omp parallel for private(center, left, right)
    for (int i = En->get_lower(); i < En->get_upper(); ++i) {
        left[0] = i;
        center[0] = i;
        right[0] = i;
        for (int j = 0; j < domain_sptr->get_grid_shape()[1]; ++j) {
            left[1] = j;
            center[1] = j;
            right[1] = j;
            for (int k = 0; k < domain_sptr->get_grid_shape()[2]; ++k) {
                left[2] = k;
                center[2] = k;
                right[2] = k;

                double delta;
                if (center[index] == lower_limit) {
                    right[index] = center[index] + 1;
                    delta = cell_size;
                } else if (center[index] == upper_limit - 1) {
                    left[index] = center[index] - 1;
                    delta = cell_size;
                } else {
                    right[index] = center[index] + 1;
                    left[index] = center[index] - 1;
                    delta = 2.0 * cell_size;
                }
                // $\vec{E} = - \grad \phi$
                En_a(center) = -(phi_a(right) - phi_a(left)) / delta;
            }
        }
    }
    En->set_normalization(phi.get_normalization());
    return En;
}

Rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_global_electric_field_component_allreduce(
        Distributed_rectangular_grid const& dist_field)
{
    Rectangular_grid_sptr global_field(new Rectangular_grid(domain_sptr));

    std::memset( (void*)global_field->get_grid_points().data(), 0, 
            global_field->get_grid_points().num_elements()*sizeof(double) );

    #pragma omp parallel for
    for (int i = dist_field.get_lower();
            i < std::min(grid_shape[0], dist_field.get_upper()); ++i) {
        for (int j = 0; j < grid_shape[1]; ++j) {
            for (int k = 0; k < grid_shape[2]; ++k) {
                global_field->get_grid_points()[i][j][k] =
                        dist_field.get_grid_points()[i][j][k];
            }
        }
    }

    int error = MPI_Allreduce(MPI_IN_PLACE,
            (void*) global_field->get_grid_points().origin(),
            global_field->get_grid_points().num_elements(), MPI_DOUBLE, MPI_SUM,
            comm2_sptr->get());
    if (error != MPI_SUCCESS) {
        throw std::runtime_error(
                "MPI error in Space_charge_3d_open_hockney_eigen(MPI_Allreduce in get_global_electric_field_component_allreduce)");
    }
    global_field->set_normalization(dist_field.get_normalization());
    return global_field;
}

Rectangular_grid_sptr
Space_charge_3d_open_hockney_eigen::get_global_electric_field_component(
        Distributed_rectangular_grid const& dist_field)
{
     return get_global_electric_field_component_allreduce(dist_field);
}

void
Space_charge_3d_open_hockney_eigen::apply_kick(Bunch & bunch,
        Rectangular_grid const& En, double delta_t, int component)
{
// $\delta \vec{p} = \vec{F} \delta t = q \vec{E} \delta t$
    double q = bunch.get_reference_particle().get_charge() * pconstants::e; // [C]
// delta_t_beam: [s] in beam frame
    double delta_t_beam = delta_t / bunch.get_reference_particle().get_gamma();
// unit_conversion: [kg m/s] to [Gev/c]
    double unit_conversion = pconstants::c / (1.0e9 * pconstants::e);
// scaled p = p/p_ref
    double p_scale = 1.0 / bunch.get_reference_particle().get_momentum();
    double factor = unit_conversion * q * delta_t_beam * En.get_normalization()
            * p_scale;

    int ps_component = 2 * component + 1;
    Rectangular_grid_domain & domain(*En.get_domain_sptr());
    MArray3d_ref grid_points(En.get_grid_points());

    #pragma omp parallel for
    for (int part = 0; part < bunch.get_local_num(); ++part) {
        double x = bunch.get_local_particles()(part, Bunch::x);
        double y = bunch.get_local_particles()(part, Bunch::y);
        double z = bunch.get_local_particles()(part, Bunch::z);
        double grid_val = interpolate_rectangular_zyx(x, y, z, domain,
                grid_points);
        bunch.get_local_particles()(part, ps_component) += factor * grid_val;
    }
}

void
Space_charge_3d_open_hockney_eigen::apply(Bunch & bunch, double time_step,
        int verbosity)
{
    double t = simple_timer_current();
    setup_communication(bunch.get_comm_sptr());
    int comm_compare;
    t = simple_timer_show(t, "sc-setup-communication");
//    bunch.convert_to_state(Bunch::fixed_t_bunch);
    t = simple_timer_show(t, "sc-convert-to-state");
    Rectangular_grid_sptr local_rho(get_local_charge_density(bunch)); // [C/m^3]
    t = simple_timer_show(t, "sc-get-local-rho");
    Distributed_rectangular_grid_sptr rho2(
            get_global_charge_density2(*local_rho, bunch.get_comm_sptr())); // [C/m^3]
    t = simple_timer_show(t, "sc-get-global-rho");
    local_rho.reset();
    Distributed_rectangular_grid_sptr G2; // [1/m]
    G2 = get_green_fn2_pointlike();
    t = simple_timer_show(t, "sc-get-green-fn");
    Distributed_rectangular_grid_sptr phi2(get_scalar_field2(*rho2, *G2)); // [V]
    t = simple_timer_show(t, "sc-get-phi2");
    rho2.reset();
    G2.reset();
    Distributed_rectangular_grid_sptr phi(extract_scalar_field(*phi2));
    t = simple_timer_show(t, "sc-get-phi");
//    bunch.periodic_sort(Bunch::z);
    t = simple_timer_show(t, "sc-sort");
    phi2.reset();
    int max_component;
    max_component = 2;
    for (int component = 0; component < max_component; ++component) {
        Distributed_rectangular_grid_sptr local_En(
                get_electric_field_component(*phi, component)); // [V/m]
        t = simple_timer_show(t, "sc-get-local-en");
        Rectangular_grid_sptr En(
                get_global_electric_field_component(*local_En)); // [V/m]
        t = simple_timer_show(t, "sc-get-global-en");
        apply_kick(bunch, *En, time_step, component);
        t = simple_timer_show(t, "sc-apply-kick");
    }
}

Space_charge_3d_open_hockney_eigen::~Space_charge_3d_open_hockney_eigen()
{
}
