#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "bunch.h"
#include "space_charge_3d_open_hockney.h"

const int particles_per_rank = 100000;
const double real_particles = 1.0e12;
const int ncells = 64;

void
apply_space_charge_3d_open_hockney(Bunch& bunch,
                                   std::vector<int> const& grid_shape)
{
    Space_charge_3d_open_hockney space_charge(grid_shape);
    const double time_step = 1.0e-6;
    const int verbosity = 99;
    space_charge.apply(bunch, time_step, verbosity);
}

double
do_timing(void (*applicator)(Bunch&, std::vector<int> const&), const char* name,
          Bunch& bunch, double reference_timing, const int rank)
{
    double t = 0;
    const int num_runs = 5;
    double best_time = 1e10;
    std::vector<double> times(num_runs);
    std::vector<int> grid_shape(3);
    grid_shape[0] = grid_shape[1] = grid_shape[2] = ncells;
    for (int i = 0; i < num_runs; ++i) {
        double t0 = MPI_Wtime();
        (*applicator)(bunch, grid_shape);
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
run_check(void (*applicator)(Bunch&, std::vector<int> const&), const char* name,
          int size, int rank)
{
    const double tolerance = 1.0e-14;
    const int num_test = 104 * size;
    Bunch b1(num_test * size, real_particles, size, rank);
    Bunch b2(num_test * size, real_particles, size, rank);
    std::vector<int> grid_shape(3);
    grid_shape[0] = grid_shape[1] = grid_shape[2] = ncells;
    apply_space_charge_3d_open_hockney(b1, grid_shape);
    applicator(b2, grid_shape);
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
    double reference_timing = do_timing(&apply_space_charge_3d_open_hockney,
                                        "orig", bunch, 0.0, rank);
    run_check(&apply_space_charge_3d_open_hockney, "sanity", size, rank);
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
