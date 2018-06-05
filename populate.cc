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

    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();

    std::seed_seq seed{ 11, 13, 17, 19, 23 };
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (Eigen::Index index = 0; index < 6; ++index) {
        for (Eigen::Index part = 0; part < local_num; ++part) {
            particles(part, index) = distribution(generator);
        }
        particles(6, index) = index;
    }

    auto particles6(particles.block(0, 0, local_num, 6));
    particles6.rowwise() -= particles6.colwise().mean();
    auto X((particles6.adjoint() * particles6) / particles6.rows());
    std::cout << "covariances = \n" << X << std::endl;
    Eigen::Matrix<double, 6, 6> H(X.llt().matrixL());
    auto A(H.inverse());

    auto start = std::chrono::high_resolution_clock::now();
//    particles6.transpose() =
//        H.colPivHouseholderQr().solve(particles6.transpose());
    particles6 *= A.transpose();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "adjustment took " << elapsed_seconds.count() << "seconds\n";

    auto Xfinal((particles6.adjoint() * particles6) / particles6.rows());
    std::cout << "final covariances = \n" << Xfinal << std::endl;

    bunch.write_particle_matrix("populated.dat");

    std::cout << "wrote populated.dat\n";
    return 0;
}
