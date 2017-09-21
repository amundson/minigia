#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "bunch.h"
#include "gsvector.h"

const int particles_per_rank = 100000;
const double real_particles = 1.0e12;

const double dummy_length = 2.1;
const double dummy_reference_time = 0.345;

class drift
{
public:
    drift() {}
    double Length() const { return dummy_length; }
    double getReferenceTime() const { return dummy_reference_time; }
};

inline double
invsqrt(double x)
{
    return 1.0 / sqrt(x);
}

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
    Bunch::Particles & particles = bunch.get_local_particles();
    double length = thedrift.Length();
    double reference_momentum = bunch.get_reference_particle().get_momentum();
    double m = bunch.get_mass();
    double reference_time = thedrift.getReferenceTime();

    for (int part = 0; part < local_num; ++part) {
        double dpop(particles(part, Bunch::dpop));
        double xp(particles(part, Bunch::xp));
        double yp(particles(part, Bunch::yp));
        double inv_npz =
            1.0 / sqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
        double lxpr = xp * length * inv_npz;
        double lypr = yp * length * inv_npz;
        double D = sqrt(lxpr * lxpr + length * length + lypr * lypr);
        double p = reference_momentum + dpop * reference_momentum;
        double E = sqrt(p * p + m * m);
        double beta = p / E;
        double x(particles(part, Bunch::x));
        double y(particles(part, Bunch::y));
        double cdt(particles(part, Bunch::cdt));
        x += lxpr;
        y += lypr;
        cdt += D / beta - reference_time;
        particles(part, Bunch::x) = x;
        particles(part, Bunch::y) = y;
        particles(part, Bunch::cdt) = cdt;
    }
}

void
propagate_double(Bunch& bunch, drift& thedrift)
{
    int local_num = bunch.get_local_num();
    const double length = thedrift.Length();
    const double reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const double m = bunch.get_mass();
    const double reference_time = thedrift.getReferenceTime();
    double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa,
        *RESTRICT cdta, *RESTRICT dpopa;
    bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

    for (int part = 0; part < local_num; ++part) {
        double x(xa[part]);
        double xp(xpa[part]);
        double y(ya[part]);
        double yp(ypa[part]);
        double cdt(cdta[part]);
        double dpop(dpopa[part]);

        drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m,
                   reference_time);

        xa[part] = x;
        ya[part] = y;
        cdta[part] = cdt;
    }
}

void
propagate_gsv(Bunch& bunch, drift& thedrift)
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
    for (size_t i = 0; i < num_runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        (*propagator)(bunch, thedrift);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto time =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
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
    const double real_num = 1.0e12;
    Bunch b1(num_test * size, real_num, size, rank);
    Bunch b2(num_test * size, real_num, size, rank);
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

    Bunch bunch(size * particles_per_rank, real_particles, size, rank);
    drift thedrift;

    double reference_timing =
        do_timing(&propagate_orig, "orig", bunch, thedrift, 0.0, rank);

    run_check(&propagate_double, "optimized", thedrift, size, rank);
    double opt_timing = do_timing(&propagate_double, "optimized", bunch,
                                  thedrift, reference_timing, rank);

    if (rank == 0) {
        std::cout << "GSVector::implementation = " << GSVector::implementation
                  << std::endl;
    }
    run_check(&propagate_gsv, "vectorized", thedrift, size, rank);
    do_timing(&propagate_gsv, "vectorized", bunch, thedrift, opt_timing,
              rank);
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
