#ifndef POPULATE_H
#define POPULATE_H
#include "bunch.h"
#include <random>

void
fill_random_gaussian(Bunch& bunch)
{
    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
    // primes number 1e6, 2e6, 3e6, 4e6, 5e6
    std::seed_seq seed{ 15485863, 32452843, 49979687, 67867967, 86028121 };
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (Eigen::Index part = 0; part < local_num; ++part) {
        for (Eigen::Index index = 0; index < 6; ++index) {
            particles(part, index) = distribution(generator);
        }
    }
}

void
force_unit_covariance(Bunch& bunch)
{
    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
    auto particles6(particles.block(0, 0, local_num, 6));

    particles6.rowwise() -= particles6.colwise().mean();
    auto X((particles6.adjoint() * particles6) / particles6.rows());
    Eigen::Matrix<double, 6, 6> H(X.llt().matrixL());
    auto A(H.inverse());
    particles6 *= A.transpose();
}

void
set_covariance(Bunch& bunch, Eigen::Matrix<double, 6, 6> const& covariance)
{
    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
    auto particles6(particles.block(0, 0, local_num, 6));
    Eigen::Matrix<double, 6, 6> G(covariance.llt().matrixL());
    particles6 *= G.transpose();
}

// Second moments found at beginning of first space charge step
// in fodo_space_charge
Eigen::Matrix<double, 6, 6>
example_mom2()
{
    Eigen::Matrix<double, 6, 6> mom2;
    mom2(0, 0) = 3.219e-05;
    mom2(0, 1) = 1.141e-06;
    mom2(0, 2) = 0;
    mom2(0, 3) = 0;
    mom2(0, 4) = 0;
    mom2(0, 5) = 0;
    mom2(1, 0) = 1.141e-06;
    mom2(1, 1) = 7.152e-08;
    mom2(1, 2) = 0;
    mom2(1, 3) = 0;
    mom2(1, 4) = 0;
    mom2(1, 5) = 0;
    mom2(2, 0) = 0;
    mom2(2, 1) = 0;
    mom2(2, 2) = 7.058e-06;
    mom2(2, 3) = -3.226e-07;
    mom2(2, 4) = 0;
    mom2(2, 5) = 0;
    mom2(3, 0) = 0;
    mom2(3, 1) = 0;
    mom2(3, 2) = -3.226e-07;
    mom2(3, 3) = 1.564e-07;
    mom2(3, 4) = 0;
    mom2(3, 5) = 0;
    mom2(4, 0) = 0;
    mom2(4, 1) = 0;
    mom2(4, 2) = 0;
    mom2(4, 3) = 0;
    mom2(4, 4) = 0.0001643;
    mom2(4, 5) = -2.507e-09;
    mom2(5, 0) = 0;
    mom2(5, 1) = 0;
    mom2(5, 2) = 0;
    mom2(5, 3) = 0;
    mom2(5, 4) = -2.507e-09;
    mom2(5, 5) = 1e-08;
    return mom2;
}

void
set_particle_ids(Bunch& bunch)
{
    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
    for (Eigen::Index part = 0; part < local_num; ++part) {
        particles(part, 6) = part;
    }
}

void
show_covariance(Bunch const& bunch)
{
    Bunch::Particles const& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();
    auto particles6(particles.block(0, 0, local_num, 6));
    auto X((particles6.adjoint() * particles6) / particles6.rows());
    std::cout << "covariance =\n" << X << std::endl;
}

void
populate_gaussian(Bunch& bunch)
{
    fill_random_gaussian(bunch);
    force_unit_covariance(bunch);
    set_covariance(bunch, example_mom2());
    set_particle_ids(bunch);
}
#endif // POPULATE_H
