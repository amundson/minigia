#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "bunch.h"
#include "bunch_data_paths.h"
#include "gsvector.h"
#include "populate.h"

#ifdef HAVE_CHEF
#include <beamline/ParticleBunch.h>
#include <beamline/drift.h>
#endif

const int particles_per_rank = 100000;
const double real_particles = 1.0e12;

const double dummy_length = 2.1;
const double dummy_reference_time = 0.345;

class libff_drift
{
public:
    libff_drift() {}
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
libff_drift_unit(T& x, T& y, T& cdt, T& xp, T& yp, T& dpop, double length,
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

#ifdef HAVE_CHEF
void
propagate_chef(ParticleBunch& chef_bunch, drift& drift)
{
    drift.propagate(chef_bunch);
}
#endif

void
propagate_orig(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    Bunch::Particles& particles = bunch.get_local_particles();
    auto length = thelibff_drift.Length();
    auto reference_momentum = bunch.get_reference_particle().get_momentum();
    auto m = bunch.get_mass();
    auto reference_time = thelibff_drift.getReferenceTime();

    for (Eigen::Index part = 0; part < local_num; ++part) {
        auto dpop(particles(part, Bunch::dpop));
        auto xp(particles(part, Bunch::xp));
        auto yp(particles(part, Bunch::yp));
        auto inv_npz =
            1.0 / sqrt((dpop + 1.0) * (dpop + 1.0) - xp * xp - yp * yp);
        auto lxpr = xp * length * inv_npz;
        auto lypr = yp * length * inv_npz;
        auto D = sqrt(lxpr * lxpr + length * length + lypr * lypr);
        auto p = reference_momentum + dpop * reference_momentum;
        auto E = sqrt(p * p + m * m);
        auto beta = p / E;
        auto x(particles(part, Bunch::x));
        auto y(particles(part, Bunch::y));
        auto cdt(particles(part, Bunch::cdt));
        x += lxpr;
        y += lypr;
        cdt += D / beta - reference_time;
        particles(part, Bunch::x) = x;
        particles(part, Bunch::y) = y;
        particles(part, Bunch::cdt) = cdt;
    }
}

void
propagate_double(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();
    double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa,
        *RESTRICT cdta, *RESTRICT dpopa;
    bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

    for (int part = 0; part < local_num; ++part) {
        auto x(xa[part]);
        auto xp(xpa[part]);
        auto y(ya[part]);
        auto yp(ypa[part]);
        auto cdt(cdta[part]);
        auto dpop(dpopa[part]);

        libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m,
                         reference_time);

        xa[part] = x;
        ya[part] = y;
        cdta[part] = cdt;
    }
}

#if defined(_OPENMP)
void
propagate_omp_simd(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();
    double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa,
        *RESTRICT cdta, *RESTRICT dpopa;
    bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

#pragma omp simd
    for (int part = 0; part < local_num; ++part) {
        auto x(xa[part]);
        auto xp(xpa[part]);
        auto y(ya[part]);
        auto yp(ypa[part]);
        auto cdt(cdta[part]);
        auto dpop(dpopa[part]);

        libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m,
                         reference_time);

        xa[part] = x;
        ya[part] = y;
        cdta[part] = cdt;
    }
}

void
propagate_omp_simd2(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();
    double *RESTRICT xa, *RESTRICT xpa, *RESTRICT ya, *RESTRICT ypa,
        *RESTRICT cdta, *RESTRICT dpopa;
    bunch.set_arrays(xa, xpa, ya, ypa, cdta, dpopa);

#pragma omp simd
    for (int part = 0; part < local_num; ++part) {
        libff_drift_unit(xa[part], ya[part], cdta[part], xpa[part], ypa[part],
                         dpopa[part], length, reference_momentum, m,
                         reference_time);
    }
}

void
propagate_omp_simd3(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();
    auto& particles(bunch.get_local_particles());

#pragma omp simd
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}

void
propagate_omp_simd3_2(Bunch& bunch, libff_drift& thelibff_drift)
{
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();

    auto& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(2)
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}

void
propagate_omp_simd3_4(Bunch& bunch, libff_drift& thelibff_drift)
{
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();

    auto& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(4)
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}

void
propagate_omp_simd3_8(Bunch& bunch, libff_drift& thelibff_drift)
{
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();

    auto& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(8)
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}

void
propagate_omp_simd3_16(Bunch& bunch, libff_drift& thelibff_drift)
{
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();

    auto& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(16)
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}
void
propagate_omp_simd3_32(Bunch& bunch, libff_drift& thelibff_drift)
{
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();

    auto& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
#pragma omp parallel for simd num_threads(32)
    for (Eigen::Index part = 0; part < local_num; ++part) {
        libff_drift_unit(particles(part, Bunch::x), particles(part, Bunch::y),
                         particles(part, Bunch::cdt),
                         particles(part, Bunch::xp), particles(part, Bunch::yp),
                         particles(part, Bunch::dpop), length,
                         reference_momentum, m, reference_time);
    }
}
#endif

void
propagate_gsv(Bunch& bunch, libff_drift& thelibff_drift)
{
    auto local_num = bunch.get_local_num();
    if (local_num % GSVector::size != 0) {
        throw std::runtime_error(
            "local number of particles must be a multiple of GSVector::size");
    }
    const auto length = thelibff_drift.Length();
    const auto reference_momentum =
        bunch.get_reference_particle().get_momentum();
    const auto m = bunch.get_mass();
    const auto reference_time = thelibff_drift.getReferenceTime();
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

        libff_drift_unit(x, y, cdt, xp, yp, dpop, length, reference_momentum, m,
                         reference_time);

        x.store(&xa[part]);
        y.store(&ya[part]);
        cdt.store(&cdta[part]);
    }
}

#ifdef HAVE_CHEF
double
do_chef_timing(void (*propagator)(ParticleBunch&, drift&), const char* name,
               ParticleBunch& chef_bunch, drift& the_drift,
               double reference_timing, const int rank)
{
    double t = 0;
    const int num_runs = 100;
    auto best_time = std::numeric_limits<double>::max();
    std::vector<double> times(num_runs);
    for (size_t i = 0; i < num_runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        (*propagator)(chef_bunch, the_drift);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto time =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start)
                .count();
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
#endif

double
do_timing(void (*propagator)(Bunch&, libff_drift&), const char* name,
          Bunch& bunch, libff_drift& thelibff_drift, double reference_timing,
          const int rank)
{
    double t = 0;
    const int num_runs = 100;
    auto best_time = std::numeric_limits<double>::max();
    std::vector<double> times(num_runs);
    for (size_t i = 0; i < num_runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        (*propagator)(bunch, thelibff_drift);
        const auto end = std::chrono::high_resolution_clock::now();
        const auto time =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start)
                .count();
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
run_check(void (*propagator)(Bunch&, libff_drift&), const char* name,
          libff_drift& thelibff_drift, int size, int rank)
{
    const double tolerance = 1.0e-14;
    const int num_test = 104 * size;
    const double real_num = 1.0e12;
    Bunch b1(num_test * size, real_num, size, rank);
    Bunch b2(num_test * size, real_num, size, rank);
    propagate_orig(b1, thelibff_drift);
    propagator(b2, thelibff_drift);
    if (!check_equal(b1, b2, tolerance)) {
        std::cerr << "run_check failed for " << name << std::endl;
    }
}

#ifdef HAVE_CHEF
Particle
reference_particle_to_chef_particle(
    Reference_particle const& reference_particle)
{
    double mass = reference_particle.get_mass();
    double momentum = reference_particle.get_momentum();
    int charge = reference_particle.get_charge();
    const double mass_tolerance = 1.0e-5;
    Vector chef_state(6);
    chef_state[0] = 0;
    chef_state[1] = 0;
    chef_state[2] = 0;
    chef_state[3] = 0;
    chef_state[4] = 0;
    chef_state[5] = 0;
    if (charge == 1) {
        if (floating_point_equal(mass, PH_NORM_mp, mass_tolerance)) {
            Proton proton;
            proton.SetReferenceMomentum(momentum);
            proton.State() = chef_state;
            return proton;
        } else {
            if (floating_point_equal(mass, PH_NORM_me, mass_tolerance)) {
                Positron positron;
                positron.SetReferenceMomentum(momentum);
                positron.State() = chef_state;
                return positron;
            } else {
                if (floating_point_equal(mass, PH_NORM_mmu, mass_tolerance)) {
                    AntiMuon antimuon;
                    antimuon.SetReferenceMomentum(momentum);
                    antimuon.State() = chef_state;
                    return antimuon;
                } else {
                    throw(runtime_error(
                        "reference_particle_to_chef_particle: particle mass "
                        "not equal to proton, electron or muon mass."));
                }
            }
        }
    } else {
        if (charge == -1) {
            if (floating_point_equal(mass, PH_NORM_mp, mass_tolerance)) {
                AntiProton antiproton;
                antiproton.SetReferenceMomentum(momentum);
                antiproton.State() = chef_state;
                return antiproton;
            } else {
                if (floating_point_equal(mass, PH_NORM_me, mass_tolerance)) {
                    Electron electron;
                    electron.SetReferenceMomentum(momentum);
                    electron.State() = chef_state;
                    return electron;
                } else {
                    if (floating_point_equal(mass, PH_NORM_mmu,
                                             mass_tolerance)) {
                        Muon muon;
                        muon.SetReferenceMomentum(momentum);
                        muon.State() = chef_state;
                        return muon;
                    } else {
                        throw(
                            runtime_error("reference_particle_to_chef_particle:"
                                          " particle mass not equal to proton, "
                                          "electron or muon mass."));
                    }
                }
            }
        } else {
            throw(runtime_error("reference_particle_to_chef_particle: particle "
                                "does not have unit charge."));
        }
    }
}
#endif

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

    Bunch bunch(particles_per_rank, real_particles, 1, 0);
    populate_gaussian(bunch);
    libff_drift thelibff_drift;

#ifdef HAVE_CHEF
    ParticleBunch chef_bunch(
        reference_particle_to_chef_particle(bunch.get_reference_particle()));

    auto particles = bunch.get_local_particles();
    auto local_num = bunch.get_local_num();
    for (int part = 0; part < local_num; ++part) {
        Particle chef_particle(reference_particle_to_chef_particle(
            bunch.get_reference_particle()));
        chef_particle.set_x(particles(part, Bunch::x));
        chef_particle.set_npx(particles(part, Bunch::xp));
        chef_particle.set_y(particles(part, Bunch::y));
        chef_particle.set_npy(particles(part, Bunch::yp));
        chef_particle.set_cdt(particles(part, Bunch::cdt));
        chef_particle.set_ndp(particles(part, Bunch::dpop));
        chef_bunch.append(chef_particle);
    }
    drift the_drift("drift", dummy_length);
#endif

    auto reference_timing =
        do_timing(&propagate_orig, "orig", bunch, thelibff_drift, 0.0, rank);

#ifdef HAVE_CHEF
    do_chef_timing(&propagate_chef, "chef", chef_bunch, the_drift,
                   reference_timing, rank);
#endif

    run_check(&propagate_double, "optimized", thelibff_drift, size, rank);
    auto opt_timing = do_timing(&propagate_double, "optimized", bunch,
                                thelibff_drift, reference_timing, rank);
    if (rank == 0) {
        std::cout << "GSVector::implementation = " << GSVector::implementation
                  << std::endl;
    }
    run_check(&propagate_gsv, "vectorized", thelibff_drift, size, rank);
    do_timing(&propagate_gsv, "vectorized", bunch, thelibff_drift, opt_timing,
              rank);

#if defined(_OPENMP)
    run_check(&propagate_omp_simd, "omp simd", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd, "omp simd", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd2, "omp simd2", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd2, "omp simd2", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3, "omp simd3", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3, "omp simd3", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3_2, "omp simd3_2", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3_2, "omp simd3_2", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3_4, "omp simd3_4", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3_4, "omp simd3_4", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3_8, "omp simd3_8", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3_8, "omp simd3_8", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3_16, "omp simd3_16", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3_16, "omp simd3_16", bunch, thelibff_drift,
              opt_timing, rank);
    run_check(&propagate_omp_simd3_32, "omp simd3_32", thelibff_drift, size, rank);
    do_timing(&propagate_omp_simd3_32, "omp simd3_32", bunch, thelibff_drift,
              opt_timing, rank);
#endif
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
