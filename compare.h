#ifndef COMPARE_H
#define COMPARE_H

#include "fftwpp/Complex.h"
#include "multi_array_typedefs.h"
#include <Eigen/Dense>
#include <iostream>

typedef std::array<long, 3> Lshape_t;

inline bool
floating_point_equal(double a, double b, double tolerance)
{
    if (std::abs(a) < tolerance) {
        return (std::abs(a - b) < tolerance);
    } else {
        return (std::abs((a - b) / a) < tolerance);
    }
}

inline double
marray_check_equal(MArray3d const& a, MArray3d const& b, long lower, long upper,
                   double tolerance)
{
    //    return a.isApprox(b, tolerance);
    //    std::cerr << "marray_check_equal " << a.shape()[0] << ", " <<
    //    a.shape()[1] << ", " << a.shape()[2] << std::endl;
    double max_diff = -1;
    for (long i = lower; i < upper; ++i) {
        for (long j = 0; j < a.shape()[1]; ++j) {
            for (long k = 0; k < a.shape()[2]; ++k) {
                double diff = std::abs(a[i][j][k] - b[i][j][k]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (!floating_point_equal(a[i][j][k], b[i][j][k], tolerance)) {
                    std::cerr << "marray_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a[i][j][k] << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b[i][j][k] << std::endl;
                    std::cerr << "  a-b = " << a[i][j][k] - b[i][j][k]
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

inline std::complex<double>
marray_check_equal(MArray3dc const& a, MArray3dc const& b, long lower,
                   long upper, double tolerance)
{
    //    return a.isApprox(b, tolerance);
    //    std::cerr << "marray_check_equal " << a.shape()[0] << ", " <<
    //    a.shape()[1] << ", " << a.shape()[2] << std::endl;
    std::complex<double> max_diff(-1, -1);
    for (long i = lower; i < upper; ++i) {
        for (long j = 0; j < a.shape()[1]; ++j) {
            for (long k = 0; k < a.shape()[2]; ++k) {
                std::complex<double> diff(
                    std::abs(a[i][j][k].real() - b[i][j][k].real()),
                    std::abs(a[i][j][k].imag() - b[i][j][k].imag()));
                if (diff.real() > max_diff.real()) {
                    max_diff =
                        std::complex<double>(diff.real(), max_diff.imag());
                }
                if (diff.imag() > max_diff.imag()) {
                    max_diff =
                        std::complex<double>(max_diff.real(), diff.imag());
                }
                if ((!floating_point_equal(a[i][j][k].real(), b[i][j][k].real(),
                                           tolerance) ||
                     (!floating_point_equal(a[i][j][k].imag(),
                                            b[i][j][k].imag(), tolerance)))) {
                    std::cerr << "marray_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a[i][j][k] << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b[i][j][k] << std::endl;
                    std::cerr << "  a-b = " << a[i][j][k] - b[i][j][k]
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

template <typename shape_T, typename array_T>
bool
general_carray_check_equal(shape_T const& shape, array_T const& a,
                           array_T const& b, double tolerance)
{
    for (long i = 0; i < shape[0]; ++i) {
        for (long j = 0; j < shape[1]; ++j) {
            for (long k = 0; k < shape[2]; ++k) {
                if ((!floating_point_equal(a(i, j, k).real(), b(i, j, k).real(),
                                           tolerance) ||
                     (!floating_point_equal(a(i, j, k).imag(),
                                            b(i, j, k).imag(), tolerance)))) {
                    std::cerr << "general_array_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a(i, j, k) << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b(i, j, k) << std::endl;
                    std::cerr << "  a-b = " << a(i, j, k) - b(i, j, k)
                              << ", tolerance = " << tolerance << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

// First array is the sub array, the second is the full array
// lower and upper correspond to the portions covered by the sub array
// z dimension is assumed to be the same in both arrays
template <typename array_T1, typename array_T2>
Complex
general_subcarray_check_equal(Lshape_t const& lower, Lshape_t const& upper,
                              array_T1 const& a, array_T2 const& b,
                              double tolerance)
{
    Complex max_diff = -1;
    for (long i2 = lower[0]; i2 < upper[0]; ++i2) {
        long i1 = i2 - lower[0];
        for (long j2 = lower[1]; j2 < upper[1]; ++j2) {
            long j1 = j2 - lower[1];
            for (long k1 = 0; k1 < upper[2]; ++k1) {
                long k2 = k1;
                Complex diff(std::abs(a(i1, j1, k1).re - b(i2, j2, k2).re),
                             std::abs(a(i1, j1, k1).im - b(i2, j2, k2).im));
                if (diff.re > max_diff.re) {
                    max_diff = Complex(diff.re, max_diff.im);
                }
                if (diff.im > max_diff.im) {
                    max_diff = Complex(max_diff.re, diff.im);
                }
                if ((!floating_point_equal(a(i1, j1, k1).re, b(i2, j2, k2).re,
                                           tolerance) ||
                     (!floating_point_equal(a(i1, j1, k1).im, b(i2, j2, k2).im,
                                            tolerance)))) {
                    std::cerr << "general_subcarray_check_equal:\n";
                    std::cerr << "  a(" << i1 << "," << j1 << "," << k1
                              << ") = " << a(i1, j1, k1) << std::endl;
                    std::cerr << "  b(" << i2 << "," << j2 << "," << k2
                              << ") = " << b(i2, j2, k2) << std::endl;
                    std::cerr << "  a-b = " << a(i1, j1, k1) - b(i2, j2, k2)
                              << ", tolerance = " << tolerance << std::endl;
                    //                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

template <typename shape_T, typename array_T>
double
general_subarray_check_equal(shape_T const& lower, shape_T const& upper,
                             array_T const& a, array_T const& b,
                             double tolerance)
{
    double max_diff = -1;
    for (long ii = lower[0]; ii < upper[0]; ++ii) {
        long i = ii - lower[0];
        for (long jj = lower[1]; jj < upper[1]; ++jj) {
            long j = -lower[1];
            for (long k = 0; k < upper[2]; ++k) {
                auto diff = std::abs(a(i, j, k) - b(i, j, k));
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (!floating_point_equal(a(i, j, k), b(i, j, k), tolerance)) {
                    std::cerr << "general_array_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a(i, j, k) << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b(i, j, k) << std::endl;
                    std::cerr << "  a-b = " << a(i, j, k) - b(i, j, k)
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

template <typename shape_T, typename array_T>
double
general_array_check_equal(shape_T const& shape, array_T const& a,
                          array_T const& b, double tolerance)
{
    double max_diff = -1;
    for (long i = 0; i < shape[0]; ++i) {
        for (long j = 0; j < shape[1]; ++j) {
            for (long k = 0; k < shape[2]; ++k) {
                auto diff = std::abs(a(i, j, k) - b(i, j, k));
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (!floating_point_equal(a(i, j, k), b(i, j, k), tolerance)) {
                    std::cerr << "general_array_check_equal:\n";
                    std::cerr << "  a(" << i << "," << j << "," << k
                              << ") = " << a(i, j, k) << std::endl;
                    std::cerr << "  b(" << i << "," << j << "," << k
                              << ") = " << b(i, j, k) << std::endl;
                    std::cerr << "  a-b = " << a(i, j, k) - b(i, j, k)
                              << ", tolerance = " << tolerance << std::endl;
                    return max_diff;
                }
            }
        }
    }
    return max_diff;
}

inline bool
eigen_check_equal(Eigen::Matrix<double, Eigen::Dynamic, 7> const& a,
                  Eigen::Matrix<double, Eigen::Dynamic, 7> const& b,
                  double tolerance)
{
    //    return a.isApprox(b, tolerance);
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        for (Eigen::Index j = 0; j < a.cols(); j++) {
            if (!floating_point_equal(a(i, j), b(i, j), tolerance)) {
                std::cerr << "eigen_check_equal:\n";
                std::cerr << "  a(" << i << "," << j << ") = " << a(i, j)
                          << std::endl;
                std::cerr << "  b(" << i << "," << j << ") = " << b(i, j)
                          << std::endl;
                std::cerr << "  a-b = " << a(i, j) - b(i, j)
                          << ", tolerance = " << tolerance << std::endl;
                return false;
            }
        }
    }
    return true;
}

#endif // COMPARE_H
