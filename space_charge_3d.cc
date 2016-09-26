#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "bunch.h"
#include "gsvector.h"

const int particles_per_rank = 100000;

const double dummy_length = 2.1;
const double dummy_reference_time = 0.345;

#if 0
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
#endif

#if 0
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

    std::cout << "nothing here!\n";
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
