#include "bunch.h"
#include "sobol.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#if 0
class Sobol
{
public:
    using result_type = unsigned long long;

private:
    size_t dimension;
    long long seed;
    std::vector<double> values;
    bool first;
    unsigned long long current_index;
    unsigned current_dimension;
    int calls;

public:
    Sobol(size_t dimension, long long seed = 0)
        : dimension(dimension)
        , seed(seed)
        , values(dimension)
        , first(true)
        , current_index(0)
        , current_dimension(dimension)
        , calls(0)
    {}
    Eigen::RowVectorXd get_vector()
    {
        Eigen::RowVectorXd retval(dimension);
        //        i8_sobol(dimension, &seed, &retval[0]);
        return retval;
    }
    double get_double()
    {
        if (current_dimension >= dimension) {
            //            i8_sobol(dimension, &seed, &values[0]);
            current_dimension = 0;
            if (!first) {
                ++current_index;
            }
            first = false;
        }
        double retval(sobol::sample(current_index, current_dimension));
        std::cout << "generator(" << current_index << "," << current_dimension
                  << ") ";
        ++current_dimension;
        ++calls;
        std::cout << calls << " calls to generator\n";
        return retval;
    }
    result_type operator()()
    {
        if (current_dimension >= dimension) {
            //            i8_sobol(dimension, &seed, &values[0]);
            current_dimension = 0;
            if (!first) {
                ++current_index;
            }
            first = false;
        }
        result_type retval(
            sobol::sample_integer(current_index, current_dimension));
        std::cout << "generator(" << current_index << "," << current_dimension
                  << ") ";
        ++current_dimension;
        ++calls;
        std::cout << calls << " calls to generator\n";
        return retval;
    }
    static constexpr result_type min() { return 0.0; }
    static result_type max() { return sobol::sample_integer_max(); }
};
#endif

class Sobol3
{
public:
    using result_type = unsigned long long;

private:
    size_t dimension;
    long long seed;
    std::vector<double> values;
    bool first;
    unsigned long long current_index;
    unsigned current_dimension;
    int calls;

public:
    Sobol3(size_t dimension, long long seed = 0)
        : dimension(dimension)
        , seed(seed)
        , values(dimension)
        , current_index(0)
        , calls(0)
    {}
    Eigen::RowVectorXd get_vector()
    {
        Eigen::RowVectorXd retval(dimension);
        //        i8_sobol(dimension, &seed, &retval[0]);
        return retval;
    }
    double get_double()
    {
        double retval(sobol::sample(current_index, dimension));
        std::cout << "generator(" << current_index << "," << dimension << ") ";
        ++current_index;
        ++calls;
        std::cout << calls << " calls to generator\n";
        return retval;
    }
    result_type operator()()
    {
        result_type retval(sobol::sample_integer(current_index, dimension));
        std::cout << "generator(" << current_index << "," << dimension << ") ";
        ++current_index;
        ++calls;
        std::cout << calls << " calls to generator" << dimension << "\n";
        return retval;
    }
    static constexpr result_type min() { return 0.0; }
    static result_type max() { return sobol::sample_integer_max(); }
};

class Sobol_uniform
{
private:
    unsigned dimension;
    double scale, offset;

public:
    Sobol_uniform(unsigned dimension, double min = 0.0, double max = 1.0)
        : dimension(dimension)
        , scale(max - min)
        , offset(min)
    {}
    inline double operator()(unsigned long long index)
    {
        return scale * sobol::sample(index, dimension) + offset;
    }
};

class Sobol_normal
{
private:
    double mu, sigma;
    double last_even, last_odd;
    unsigned long long last_index;
    unsigned last_even_dimension;
    //    std::mt19937 generator;
    bool cached;

public:
    Sobol_normal(double mu = 0.0, double sigma = 1.0)
        : mu(mu)
        , sigma(sigma)
        , last_even(0.0)
        , last_odd(0.0)
        , last_index(0)
        , last_even_dimension(0)
        , cached(false)
    {
        //        std::seed_seq seed{ 11 + 29 * dimension, 13 + 31 * dimension,
        //                            17 + 37 * dimension, 19 + 41 * dimension,
        //                            23 + 43 * dimension };
        //        generator = std::mt19937(seed);
    }

    inline double operator()(unsigned dimension, unsigned long long index)
    {
        bool even = dimension % 2 == 0;
        unsigned even_dimension = even ? dimension : (dimension - 1);
            std::cout << "check: " << cached << " "
                      << last_index << " "
                      << index << " "
                      << last_even_dimension << " "
                      << even_dimension << " "
                      << dimension << "\n";
        if ((!cached) || (last_index != index) ||
            (last_even_dimension != even_dimension)) {
            constexpr unsigned sobol_offset = 1;
            double u1 = sobol::sample(index + sobol_offset, even_dimension);
            double u2 = sobol::sample(index + sobol_offset, even_dimension + 1);
            //            std::cout << "gdfs:" << generator() << ", " <<
            //            generator.max() << std::endl; double norm
            //            = 1.0/generator.max(); double u1 = generator()*norm;
            //            double u2 = generator()*norm;
            constexpr double pi =
                3.1415926535897932384626433832795028841971693993751;
            last_even =
                sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * pi * u2) +
                mu;
            last_odd =
                sigma * std::sqrt(-2.0 * std::log(u1)) * std::sin(2 * pi * u2) +
                mu;
            std::cout << "wtf: " << mu << ", " << sigma << ", " << u1 << ", "
                      << u2 << std::endl;
            cached = true;
            last_even_dimension = even_dimension;
            last_index = index;
        } else {
            std::cout << "no! stop! never cached!!!!!" << (!cached) << " "
                      << (last_index != index) << " "
                      << (last_even_dimension = even_dimension) << "\n";
        }
        return even ? last_even : last_odd;
    }
};
const long total_num = 100000;
const double real_num = 1.0e12;

int
main()
{
    Bunch bunch(total_num, real_num, 1, 0);

    Bunch::Particles& particles(bunch.get_local_particles());
    auto local_num = bunch.get_local_num();

    std::seed_seq seed{ 11, 13, 17, 19, 23 };
    //    Sobol sobol(6);
    //    std::normal_distribution<double> distribution(0.0, 1.0);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
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
    //    typedef Sobol_normal My_sobol;
    //    My_sobol generator[6] = { My_sobol(0), My_sobol(1), My_sobol(2),
    //                              My_sobol(3), My_sobol(4), My_sobol(5) };
    Sobol_normal generator(0.0, 1.0);
    for (Eigen::Index index = 0; index < 6; ++index) {
        for (Eigen::Index part = 0; part < local_num; ++part) {
            //            particles(part, index) =
            //            distribution(generator[index]);
            particles(part, index) = generator(index, part);
            std::cout << part << ", " << index << ": "
                      << particles(part, index);
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
