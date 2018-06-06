#include "bunch.h"
#include "sobol.h"
#include <array>
#include <chrono>
#include <iostream>
#include <random>

class Sobol
{
public:
    using result_type = double;

private:
    size_t dimension;
    long long seed;
    std::vector<result_type> values;
    size_t current_index;
    int calls;

public:
    Sobol(size_t dimension, long long seed = 0)
        : dimension(dimension)
        , seed(seed)
        , values(dimension)
        , current_index(dimension)
        , calls(0)
    {}
    Eigen::RowVectorXd get_vector()
    {
        Eigen::RowVectorXd retval(dimension);
        i8_sobol(dimension, &seed, &retval[0]);
        return retval;
    }
    result_type operator()()
    {
        if (current_index >= dimension) {
            i8_sobol(dimension, &seed, &values[0]);
            current_index = 0;
        }
        result_type retval(values[current_index]);
        ++current_index;
        std::cout << "generator(" << retval << ") ";
        ++calls;
        std::cout << calls << " calls to generator\n";
        return retval;
    }
    static constexpr result_type min() { return 0.0; }
    static constexpr result_type max() { return 1.0; }
};

const long total_num = 10;
const double real_num = 1.0e12;

int
main()
{
    Bunch bunch(total_num, real_num, 1, 0);

    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();

    std::seed_seq seed{ 11, 13, 17, 19, 23 };
    Sobol sobol(6);
    std::normal_distribution<double> distribution(0.0, 1.0);
//    std::uniform_real_distribution<double> distribution(0.0, 1.0);
#if 0
    std::mt19937 generator(seed);
    for (Eigen::Index index = 0; index < 6; ++index) {
        for (Eigen::Index part = 0; part < local_num; ++part) {
            particles(part, index) = distribution(generator);
        }
        particles(6, index) = index;
    }
#endif
#if 0
    for (Eigen::Index part = 0; part < local_num; ++part) {
        particles.block(part, 0, 1, 6) = sobol.get_vector();
    }
#endif
#if 1
    Sobol generator(6);
    for (Eigen::Index part = 0; part < local_num; ++part) {
        for (Eigen::Index index = 0; index < 6; ++index) {
            std::cout << part << ", " << index << ": ";
            particles(part, index) = distribution(generator);
//                        particles(part, index) = generator();
            std::cout << std::endl;
        }
        //        particles(6, index) = index;
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
