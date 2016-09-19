#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "bunch.h"
#include "gsvector.h"

const int particles_per_rank = 100000;

const double dummy_length = 2.1;
const double dummy_reference_time = 0.345;

class drift
{
public:
    drift() {}
    double Length() const { return dummy_length; }
    double getReferenceTime() const { return dummy_reference_time; }
};

template <typename T>
inline void
drift_unit(T& x, T& y, T& cdt, T& xp, T& yp, T& dpop, double length,
           double reference_momentum, double m, double reference_time)
{
    T inv_npz = invsqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
    T lxpr = xp * length * inv_npz;
    T lypr = yp * length * inv_npz;
    T D2 = lxpr * lxpr + length * length + lypr * lypr;
    T p = dpop * reference_momentum + reference_momentum;
    T E2 = p * p + m * m;
    T inv_beta2 = E2 / (p * p);
    x += lxpr;
    y += lypr;
    cdt += sqrt(D2 * inv_beta2) - reference_time;
}

void
propagate_orig(Bunch& bunch, drift& thedrift)
{
    int local_num = bunch.get_local_num();
    if (local_num % 1 != 0) {
        throw std::runtime_error(
            "local number of particles must be a multiple of 4");
    }
    MArray2d_ref particles = bunch.get_local_particles();
    double length = thedrift.Length();
    double reference_momentum = bunch.get_reference_particle().get_momentum();
    double m = bunch.get_mass();
    double reference_time = thedrift.getReferenceTime();

    for (int part = 0; part < local_num; ++part) {
        double dpop(particles[part][Bunch::dpop]);
        double xp(particles[part][Bunch::xp]);
        double yp(particles[part][Bunch::yp]);
        double inv_npz =
            1.0 / sqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
        double lxpr = xp * length * inv_npz;
        double lypr = yp * length * inv_npz;
        double D = sqrt(lxpr * lxpr + length * length + lypr * lypr);
        double p = reference_momentum + dpop * reference_momentum;
        double E = sqrt(p * p + m * m);
        double beta = p / E;
        double x(particles[part][Bunch::x]);
        double y(particles[part][Bunch::y]);
        double cdt(particles[part][Bunch::cdt]);
        x += lxpr;
        y += lypr;
        cdt += D / beta - reference_time;
        particles[part][Bunch::x] = x;
        particles[part][Bunch::y] = y;
        particles[part][Bunch::cdt] = cdt;
    }
}

void
propagate(Bunch& bunch, drift& thedrift)
{
    int local_num = bunch.get_local_num();
    if (local_num % GSVector::size != 0) {
        throw std::runtime_error(
            "local number of particles must be a multiple of GSVector::size");
    }
    const double length = thedrift.Length();
    const double reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const double m = bunch.get_mass();
    const double reference_time = thedrift.getReferenceTime();
    double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa,
        *RESTRICT cdta, *RESTRICT dpopa;
    bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

    for (int part = 0; part < local_num; part += GSVector::size) {
        GSVector x(&xa[part]);
        GSVector xp(&xpa[part]);
        GSVector y(&ya[part]);
        GSVector yp(&ypa[part]);
        GSVector cdt(&cdta[part]);
        GSVector dpop(&dpopa[part]);

        drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m,
                   reference_time);

        x.store(&xa[part]);
        y.store(&ya[part]);
        cdt.store(&cdta[part]);
    }
}

double
do_timing(void (*propagator)(Bunch&, drift&), const char* name, Bunch& bunch,
          drift& thedrift, double reference_timing, const int rank)
{
    double t = 0;
    const int num_runs = 100;
    double best_time = 1e10;
    std::vector<double> times(num_runs);
    for (int i = 0; i < num_runs; ++i) {
        double t0 = MPI_Wtime();
        (*propagator)(bunch, thedrift);
        double t1 = MPI_Wtime();
        double time = t1 - t0;
        if (time < best_time) {
            best_time = time;
        }
        times.at(i) = time;
        t += time;
    }
    if (rank == 0) {
        std::cout << name << " best time = " << best_time << std::endl;
    }
    if (reference_timing > 0.0) {
        if (rank == 0) {
            std::cout << name << " speedup = " << reference_timing / best_time
                      << std::endl;
        }
    }
    return best_time;
}

void
run_check(void (*propagator)(Bunch&, drift&), const char* name, drift& thedrift,
          int size, int rank)
{
    const double tolerance = 1.0e-14;
    const int num_test = 104 * size;
    Bunch b1(num_test * size, size, rank);
    Bunch b2(num_test * size, size, rank);
    propagate_orig(b1, thedrift);
    propagator(b2, thedrift);
    if (!check_equal(b1, b2, tolerance)) {
        std::cerr << "run_check failed for " << name << std::endl;
    }
}

void
run()
{
    int error, rank, size;
    error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (error) {
        std::cerr << "MPI error" << std::endl;
        exit(error);
    }

    Bunch bunch(size * particles_per_rank, size, rank);
    drift thedrift;

    double reference_timing =
        do_timing(&propagate_orig, "orig", bunch, thedrift, 0.0, rank);

    if (rank == 0) {
        std::cout << "GSVector::implementation = " << GSVector::implementation
                  << std::endl;
    }
    run_check(&propagate, "optimized", thedrift, size, rank);
    do_timing(&propagate, "optimized", bunch, thedrift,
              reference_timing, rank);
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
