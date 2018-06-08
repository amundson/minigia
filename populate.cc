#include "bunch.h"
#include "sobol.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

const long total_num = 1000;
const double real_num = 1.0e12;

class Sobol_uniform_distribution
{
private:
    double scale, offset;

public:
    Sobol_uniform_distribution(double min = 0.0, double max = 1.0)
        : scale(max - min)
        , offset(min)
    {}
    inline double operator()(unsigned dimension, unsigned long long index)
    {
        return scale * sobol::sample(index, dimension) + offset;
    }
};

class Sobol_normal_distribution
{
private:
    double mu, sigma;
    double last_even, last_odd;
    unsigned long long last_index;
    unsigned last_even_dimension;
    bool cached;

public:
    Sobol_normal_distribution(double mu = 0.0, double sigma = 1.0)
        : mu(mu)
        , sigma(sigma)
        , last_even(0.0)
        , last_odd(0.0)
        , last_index(0)
        , last_even_dimension(0)
        , cached(false)
    {}

    inline double operator()(unsigned dimension, unsigned long long index)
    {
        bool even = dimension % 2 == 0;
        unsigned even_dimension = even ? dimension : (dimension - 1);
        if ((!cached) || (last_index != index) ||
            (last_even_dimension != even_dimension)) {
            constexpr unsigned sobol_offset = 1;
            double u1 = sobol::sample(index + sobol_offset, even_dimension);
            double u2 = sobol::sample(index + sobol_offset, even_dimension + 1);
            constexpr double pi =
                3.1415926535897932384626433832795028841971693993751;
            last_even =
                sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * pi * u2) +
                mu;
            last_odd =
                sigma * std::sqrt(-2.0 * std::log(u1)) * std::sin(2 * pi * u2) +
                mu;
            cached = true;
            last_even_dimension = even_dimension;
            last_index = index;
        }
        return even ? last_even : last_odd;
    }
};

int
main()
{
    Bunch bunch(total_num, real_num, 1, 0);

    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();

    std::seed_seq seed{ 11, 13, 17, 19, 23 };
    //    std::normal_distribution<double> distribution(0.0, 1.0);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
#if 0
    std::mt19937 generator(seed);
    for (Eigen::Index part = 0; part < local_num; ++part) {
        for (Eigen::Index index = 0; index < 6; ++index) {
            particles(part, index) = distribution(generator);
        }
        particles(part, 6) = part;
    }
#endif
#if 1
    //    Sobol_normal_distribution generator(0.0, 1.0);
    Sobol_uniform_distribution generator(0.0, 1.0);
    for (Eigen::Index part = 0; part < local_num; ++part) {
        for (Eigen::Index index = 0; index < 6; ++index) {
            particles(part, index) = generator(index, part);
        }
        particles(part, 6) = part;
    }
#endif
#if 0
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
#endif

    bunch.write_particle_matrix("populated.dat");

    std::cout << "wrote populated.dat\n";
    return 0;
}
