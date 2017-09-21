#ifndef BUNCH_H_
#define BUNCH_H_

#include <iostream>
#include <Eigen/Dense>
#include "multi_array_typedefs.h"
#include "reference_particle.h"
#include "commxx.h"
#include "restrict_extension.h"

//  J. Beringer et al. (Particle Data Group), PR D86, 010001 (2012) and 2013
//  partial update for the 2014 edition (URL: http://pdg.lbl.gov)
static const double proton_mass = 0.938272046; // Mass of proton [GeV/c^2]
static const double example_gamma = 10.0;
static const int proton_charge = 1; // Charge in units of e

class Bunch
{
public:
    static const size_t x = 0;
    static const size_t xp = 1;
    static const size_t y = 2;
    static const size_t yp = 3;
    static const size_t z = 4;
    static const size_t zp = 5;
    static const size_t cdt = 4;
    static const size_t dpop = 5;
    static const size_t id = 6;
    static const size_t particle_size = 7;

    typedef Eigen::Matrix<double, Eigen::Dynamic, 7> Particles;

    struct AView
    {
        double* RESTRICT x;
        double* RESTRICT xp;
        double* RESTRICT y;
        double* RESTRICT yp;
        double* RESTRICT cdt;
        double* RESTRICT dpop;
    };

private:
    Reference_particle reference_particle;
    double* storage;
    size_t local_num, total_num;
    double real_num;
    Particles local_particles;
    Commxx_sptr comm_sptr;
    AView aview;

public:
    Bunch(size_t total_num, double real_num, int mpi_size, int mpi_rank)
        : reference_particle(proton_charge, proton_mass,
                             example_gamma * proton_mass),
          local_num(total_num / mpi_size), // jfa FIXME!
          total_num(total_num),
          real_num(real_num),
          local_particles(local_num, 7),
          comm_sptr(new Commxx)
    {
        auto * origin = local_particles.data();
        aview.x = local_particles.col(Bunch::x).data();
        aview.xp = local_particles.col(Bunch::xp).data();
        aview.y = local_particles.col(Bunch::y).data();
        aview.yp = local_particles.col(Bunch::yp).data();
        aview.cdt = local_particles.col(Bunch::cdt).data();
        aview.dpop = local_particles.col(Bunch::dpop).data();
        for (size_t part = 0; part < local_num; ++part) {
            size_t index = part + mpi_rank * mpi_size;
            local_particles(part, Bunch::x) = 1.0e-6 * index;
            local_particles(part, Bunch::xp) = 1.1e-8 * index;
            local_particles(part, Bunch::y) = 1.3e-6 * index;
            local_particles(part, Bunch::yp) = 1.4e-8 * index;
            local_particles(part, Bunch::z) = 1.5e-4 * index;
            local_particles(part, Bunch::zp) = 1.5e-7 * index;
            local_particles(part, Bunch::id) = index;
        }
    }

    Reference_particle const& get_reference_particle() const
    {
        return reference_particle;
    }

    Particles & get_local_particles() { return local_particles; }

    Particles const& get_local_particles() const { return local_particles; }

    double get_mass() const { return reference_particle.get_mass(); }

    size_t get_local_num() const { return local_num; }

    size_t get_total_num() const { return total_num; }

    double get_real_num() const { return real_num; }

    Commxx_sptr get_comm_sptr() const { return comm_sptr; }

    void set_arrays(double* RESTRICT& xa, double* RESTRICT& xpa,
                    double* RESTRICT& ya, double* RESTRICT& ypa,
                    double* RESTRICT& cdta, double* RESTRICT& dpopa)
    {
        double* origin = local_particles.data();
        xa = local_particles.col(Bunch::x).data();
        xpa = local_particles.col(Bunch::xp).data();
        ya = local_particles.col(Bunch::y).data();
        ypa = local_particles.col(Bunch::yp).data();
        cdta = local_particles.col(Bunch::cdt).data();
        dpopa = local_particles.col(Bunch::dpop).data();
    }

    AView get_aview() { return aview; }

    virtual ~Bunch()
    {
    }
};

inline bool
floating_point_equal(double a, double b, double tolerance)
{
    if (std::abs(a) < tolerance) {
        return (std::abs(a - b) < tolerance);
    } else {
        return (std::abs((a - b) / a) < tolerance);
    }
}

inline bool
eigen_check_equal(Bunch::Particles const& a, Bunch::Particles const& b,
                        double tolerance)
{
//    return a.isApprox(b, tolerance);
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        for (Eigen::Index j = 0; j < a.cols(); j++) {
            if (!floating_point_equal(a(i, j), b(i,j), tolerance)) {
                std::cerr << "eigen_check_equal:\n";
                std::cerr << "  a(" << i << "," << j << ") = " << a(i, j)
                          << std::endl;
                std::cerr << "  b(" << i << "," << j << ") = " << b(i, j)
                          << std::endl;
                std::cerr << "  a-b = " << a(i, j) - b(i, j)
                          << ", tolerance = " << tolerance << std::endl;
                return false;
            }
        }
    }
    return true;
}

inline bool
check_equal(Bunch& b1, Bunch& b2, double tolerance)
{
    if (b1.get_local_num() != b2.get_local_num()) {
        std::cerr << "check_equal: bunch 1 has " << b1.get_local_num()
                  << "local particles, ";
        std::cerr << "bunch 2 has " << b2.get_local_num() << "local particles"
                  << std::endl;
        return false;
    }
    return eigen_check_equal(b1.get_local_particles(),
                             b2.get_local_particles(), tolerance);
}

#endif /* BUNCH_H_ */
