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
    static const int x = 0;
    static const int xp = 1;
    static const int y = 2;
    static const int yp = 3;
    static const int z = 4;
    static const int zp = 5;
    static const int cdt = 4;
    static const int dpop = 5;
    static const int id = 6;
    static const int particle_size = 7;

    typedef Eigen::Matrix<double, particle_size, Eigen::Dynamic> Particle_matrix;

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
    int local_num, total_num;
    double real_num;
    Particle_matrix local_particles;
    Commxx_sptr comm_sptr;
    AView aview;

public:
    Bunch(int total_num, double real_num, int mpi_size, int mpi_rank)
        : reference_particle(proton_charge, proton_mass,
                             example_gamma * proton_mass),
          local_num(total_num / mpi_size), // jfa FIXME!
          total_num(total_num),
          real_num(real_num),
          local_particles(particle_size, local_num),
          comm_sptr(new Commxx)
    {
        double* origin = local_particles.data();
        aview.x = origin + local_num * Bunch::x;
        aview.xp = origin + local_num * Bunch::xp;
        aview.y = origin + local_num * Bunch::y;
        aview.yp = origin + local_num * Bunch::yp;
        aview.cdt = origin + local_num * Bunch::cdt;
        aview.dpop = origin + local_num * Bunch::dpop;
        for (int part = 0; part < local_num; ++part) {
            int index = part + mpi_rank * mpi_size;
            local_particles(Bunch::x, part) = 1.0e-6 * index;
            local_particles(Bunch::xp, part) = 1.1e-8 * index;
            local_particles(Bunch::y, part) = 1.3e-6 * index;
            local_particles(Bunch::yp, part) = 1.4e-8 * index;
            local_particles(Bunch::z, part) = 1.5e-4 * index;
            local_particles(Bunch::zp, part) = 1.5e-7 * index;
            local_particles(Bunch::id, part) = index;
        }
    }

    Reference_particle const& get_reference_particle() const
    {
        return reference_particle;
    }

    Particle_matrix & get_local_particles() { return local_particles; }

    Particle_matrix const& get_local_particles() const { return local_particles; }

    double get_mass() const { return reference_particle.get_mass(); }

    int get_local_num() const { return local_num; }

    int get_total_num() const { return total_num; }

    double get_real_num() const { return real_num; }

    Commxx_sptr get_comm_sptr() const { return comm_sptr; }

    void set_arrays(double* RESTRICT& xa, double* RESTRICT& xpa,
                    double* RESTRICT& ya, double* RESTRICT& ypa,
                    double* RESTRICT& cdta, double* RESTRICT& dpopa)
    {
        double* origin = local_particles.data();
        xa = origin + local_num * Bunch::x;
        xpa = origin + local_num * Bunch::xp;
        ya = origin + local_num * Bunch::y;
        ypa = origin + local_num * Bunch::yp;
        cdta = origin + local_num * Bunch::cdt;
        dpopa = origin + local_num * Bunch::dpop;
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
multi_array_check_equal(MArray2d_ref const& a, MArray2d_ref const& b,
                        double tolerance)
{
    for (auto i = a.index_bases()[0];
         i < a.index_bases()[0] + a.shape()[0]; ++i) {
        for (auto j = a.index_bases()[1];
             j < a.index_bases()[1] + a.shape()[1]; ++j) {
            if (!floating_point_equal(a[i][j], b[i][j], tolerance)) {
                std::cerr << "multi_array_check_equal:\n";
                std::cerr << "  a[" << i << "][" << j << "] = " << a[i][j]
                          << std::endl;
                std::cerr << "  b[" << i << "][" << j << "] = " << b[i][j]
                          << std::endl;
                std::cerr << "  a-b = " << a[i][j] - b[i][j]
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
    return multi_array_check_equal(b1.get_local_particles(),
                                   b2.get_local_particles(), tolerance);
}

#endif /* BUNCH_H_ */
