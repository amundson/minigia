#include "populate.h"
#include "bunch.h"
#include <array>
#include <chrono>
#include <iostream>
#include <random>

const long total_num = 100000;
const double real_num = 1.0e12;

int
main()
{
    Bunch bunch(total_num, real_num, 1, 0);
    populate_gaussian(bunch);
    show_covariance(bunch);
    std::cout << "expected covariance =\n" << example_mom2() << std::endl;

    bunch.write_particle_matrix("populated.dat");

    std::cout << "wrote populated.dat\n";
    return 0;
}
