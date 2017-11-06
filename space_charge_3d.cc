#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "bunch.h"
#include "bunch_data_paths.h"
#include "simple_timer.h"
#include "space_charge_3d_open_hockney.h"
#include "space_charge_3d_open_hockney_eigen.h"

const int particles_per_rank = 100000;

const double dummy_length = 2.1;
const double dummy_reference_time = 0.345;

double
do_timing(Collective_operator& space_charge, Bunch const& bunch,
          double time_step, int verbosity, const char* name,
          double reference_timing)
{
    double t = 0;
    const int num_runs = 10;
    double best_time = 1e10;
    std::vector<double> times(num_runs);
    for (size_t i = 0; i < num_runs; ++i) {
        Bunch b(bunch);
        double t0 = MPI_Wtime();
        space_charge.apply(b, time_step, verbosity);
        double t1 = MPI_Wtime();
        double time = t1 - t0;
        if (time < best_time) {
            best_time = time;
        }
        times.at(i) = time;
        t += time;
    }
    //    if (rank == 0) {
    std::cout << name << " best time = " << best_time << std::endl;
    //    }
    if (reference_timing > 0.0) {
        //        if (rank == 0) {
        std::cout << name << " speedup = " << reference_timing / best_time
                  << std::endl;
        //        }
    }
    return best_time;
}

void
run_check(Collective_operator& space_charge, Collective_operator& reference,
          double time_step, int verbosity, const char* name)
{
    const double tolerance = 1.0e-14;
    Bunch b1(bunch_in_0_path);
    Bunch b2(b1);
    reference.apply(b1, time_step, verbosity);
    space_charge.apply(b2, time_step, verbosity);
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

    Bunch bunch(bunch_in_0_path);
    std::vector<int> grid_shape({ 32, 32, 128 });
    double time_step = 0.01;
    int verbosity = 99;
    Commxx_divider_sptr commxx_divider_sptr(new Commxx_divider);
    Space_charge_3d_open_hockney orig(commxx_divider_sptr, grid_shape);

    auto reference_timing =
        do_timing(orig, bunch, time_step, verbosity, "orig", 0.0);

    run_check(orig, orig, time_step, verbosity, "orig");
    do_timing(orig, bunch, time_step, verbosity, "orig", reference_timing);

    Space_charge_3d_open_hockney_eigen eigen(commxx_divider_sptr, grid_shape);
    run_check(eigen, eigen, time_step, verbosity, "eigen");
    do_timing(eigen, bunch, time_step, verbosity, "eigen", reference_timing);
}

std::ofstream global_simple_timer_out("simple_timer.out");

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
    MPI_Finalize();
    return 0;
}
