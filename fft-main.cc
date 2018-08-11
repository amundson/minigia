#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Array.h"
#include "Complex.h"
#include "compare.h"
#include "distributed_fft3d.h"
#include "fmt/format.h"

//#include "fftw++.h"
#include "align.h"
#include "mpifftw++.h"
#include "mpigroup.h"

typedef std::array<int, 3> Shape_t;

double
run_timing(const std::function<void()>& f)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    return time;
}

void
show_best_time(const std::string& label,
               const std::function<std::tuple<double, double>()>& f)
{
    const int timing_count = 10;
    auto best_time1 = std::numeric_limits<double>::max();
    auto best_time2 = std::numeric_limits<double>::max();
    auto best_total = std::numeric_limits<double>::max();
    for (int i = 0; i < timing_count; ++i) {
        double time1, time2;
        std::tie(time1, time2) = f();
        if (time1 < best_time1) {
            best_time1 = time1;
        }
        if (time2 < best_time2) {
            best_time2 = time2;
        }
        if (time2 + time1 < best_total) {
            best_total = time2 + time1;
        }
    }
    Commxx commxx;
    if (commxx.get_rank() == 0) {
        fmt::print("{} best time: {} + {} = {}\n", label, best_time1,
                   best_time2, best_total);
    }
}

template <typename T>
void
write_array3(const char* filename, T const& a)
{
    std::ofstream file(filename);
    file << a.Nx() << "\n";
    file << a.Ny() << "\n";
    file << a.Nz() << "\n";
    file.precision(16);
    for (unsigned long i = 0; i < a.Nx(); ++i) {
        for (unsigned long j = 0; j < a.Ny(); ++j) {
            for (unsigned long k = 0; k < a.Nz(); ++k) {
                file << a(i, j, k) << "\n";
            }
        }
    }
}

void
run_fftwpp()
{
    unsigned int nx = 32, ny = 16, nz = 8;
    //    unsigned int nx = 2, ny = 4, nz = 8;
    unsigned int nz_complex = nz / 2 + 1;
    //    unsigned int nz_padded = 2* nz_complex;
    const std::array<unsigned int, 3> shape{ nx, ny, nz };
    const std::array<unsigned int, 3> cshape{ nx, ny, nz_complex };
    size_t align = sizeof(Complex);
    Array::array3<double> rarray(shape[0], shape[1], shape[2], align);
    Array::array3<double> orig(shape[0], shape[1], shape[2], align);
    Array::array3<Complex> carray(cshape[0], cshape[1], cshape[2]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                orig(i, j, k) = rarray(i, j, k) = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }

    fftwpp::rcfft3d forward(shape[0], shape[1], shape[2], rarray, carray);
    auto start = std::chrono::high_resolution_clock::now();
    forward.fft(rarray, carray);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "fftwpp time = " << time << " s\n";

    write_array3("fft-carray3.dat", carray);

    fftwpp::crfft3d backward(shape[0], shape[1], shape[2], carray, rarray);
    start = std::chrono::high_resolution_clock::now();
    backward.fftNormalized(carray, rarray);
    end = std::chrono::high_resolution_clock::now();
    time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "inverse fftwpp time = " << time << " s\n";
    const double tolerance = 5.0e-11;
    auto max_diff = general_array_check_equal(shape, rarray, orig, tolerance);
    std::cout << "check roundtrip: " << max_diff << std::endl;
    //    if (max_diff > tolerance) {
    //        std::cout << "orig: " << orig << std::endl;
    //        std::cout << "rarray: " << rarray << std::endl;
    //    }
}
template <typename T>
void
write_marray(const char* filename, T const& a)
{
    std::ofstream file(filename);
    file << a.shape()[0] << "\n";
    file << a.shape()[1] << "\n";
    file << a.shape()[2] << "\n";
    file.precision(16);
    for (unsigned long i = 0; i < a.shape()[0]; ++i) {
        for (unsigned long j = 0; j < a.shape()[1]; ++j) {
            for (unsigned long k = 0; k < a.shape()[2]; ++k) {
                file << a[i][j][k] << "\n";
            }
        }
    }
}

#if FFTWPP_NOMPI
void
run_fftwpp_eigen()
{
    unsigned int nx = 32, ny = 16, nz = 8;
    //    unsigned int nx = 2, ny = 4, nz = 8;
    unsigned int nz_complex = nz / 2 + 1;
    //    unsigned int nz_padded = 2* nz_complex;
    const std::array<unsigned int, 3> shape{ nx, ny, nz };
    const std::array<unsigned int, 3> cshape{ nx, ny, nz_complex };

    Eigen::Tensor<double, 3> rarray(shape[0], shape[1], shape[2]);
    Eigen::Tensor<double, 3> orig(shape[0], shape[1], shape[2]);
    Eigen::Tensor<std::complex<double>, 3> carray(cshape[0], cshape[1],
                                                  cshape[2]);

    for (Eigen::Index i = 0; i < shape[0]; ++i) {
        for (Eigen::Index j = 0; j < shape[1]; ++j) {
            for (Eigen::Index k = 0; k < shape[2]; ++k) {
                orig(i, j, k) = rarray(i, j, k) = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }

    fftwpp::rcfft3d forward(shape[0], shape[1], shape[2], &rarray(0, 0, 0),
                            &carray(0, 0, 0));
    auto start = std::chrono::high_resolution_clock::now();
    forward.fft(&rarray(0, 0, 0), &carray(0, 0, 0));
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "eigen fftwpp time = " << time << " s\n";
}
#endif

template <typename T>
Eigen::Tensor<T, 3>
read_eigen(const char* filename)
{
    std::ifstream file(filename);
    std::array<unsigned long, 3> shape;
    file >> shape[0];
    file >> shape[1];
    file >> shape[2];
    Eigen::Tensor<T, 3> a(shape[0], shape[1], shape[2]);
    for (unsigned long i = 0; i < shape[0]; ++i) {
        for (unsigned long j = 0; j < shape[1]; ++j) {
            for (unsigned long k = 0; k < shape[2]; ++k) {
                file >> a(i, j, k);
            }
        }
    }
    return a;
}

template <typename T>
T
read_marray(const char* filename)
{
    std::ifstream file(filename);
    std::array<int, 3> cshape;
    file >> cshape[0];
    file >> cshape[1];
    file >> cshape[2];
    T a(boost::extents[cshape[0]][cshape[1]][cshape[2]]);
    for (unsigned long i = 0; i < a.shape()[0]; ++i) {
        for (unsigned long j = 0; j < a.shape()[1]; ++j) {
            for (unsigned long k = 0; k < a.shape()[2]; ++k) {
                file >> a[i][j][k];
            }
        }
    }
    return a;
}

void
old_run()
{
    const std::vector<int> shape{ 32, 16, 8 };
    //    const std::vector<int> shape{ 8, 4, 2 };
    Commxx_sptr commxx_sptr(new Commxx);
    Distributed_fft3d distributed_fft3d(shape, commxx_sptr, FFTW_ESTIMATE);
    auto lower = distributed_fft3d.get_lower();
    auto upper = distributed_fft3d.get_upper();
    std::vector<int> rshape(distributed_fft3d.get_padded_shape_real());
    MArray3d rarray(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    MArray3d orig(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    std::vector<int> cshape(distributed_fft3d.get_padded_shape_complex());
    MArray3dc carray(
        boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                orig[i][j][k] = rarray[i][j][k] = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    distributed_fft3d.transform(rarray, carray);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "time = " << time << " s\n";

    write_marray("fft-carray.dat", carray);
    auto check(read_marray<MArray3dc>("fft-carray.dat"));
    const double tolerance = 5.0e-11;
    std::cout << "check written: "
              << marray_check_equal(carray, check, lower, upper, tolerance)
              << std::endl;
    //    write_marray3dc("check.dat", check);
    start = std::chrono::high_resolution_clock::now();
    distributed_fft3d.inv_transform(carray, rarray);
    end = std::chrono::high_resolution_clock::now();
    time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "inverse time = " << time << " s\n";

    double norm = shape[0] * shape[1] * shape[2];
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                rarray[i][j][k] *= 1.0 / norm;
            }
        }
    }

    // zero out padded region
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < rshape[1]; ++j) {
            for (int k = shape[2]; k < rshape[2]; ++k) {
                rarray[i][j][k] = 0.0;
                orig[i][j][k] = 0.0;
            }
        }
    }
    //    write_marray("wtf-rarray.dat", rarray);
    //    write_marray("wtf-orig.dat", orig);
    std::cout << "check roundtrip: "
              << marray_check_equal(rarray, orig, lower, upper, tolerance)
              << std::endl;
}

std::string
carray_filename(Shape_t const& shape)
{
    return fmt::format("fft-carray_{}_{}_{}.dat", shape[0], shape[1], shape[2]);
}
void
write_check(Shape_t const& shape_in, Shape_t const& cshape_in)
{
    Commxx_sptr commxx_sptr(new Commxx);
    std::vector<int> shape({ shape_in[0], shape_in[1], shape_in[2] });
    Distributed_fft3d distributed_fft3d(shape, commxx_sptr, FFTW_ESTIMATE);
    std::vector<int> rshape(distributed_fft3d.get_padded_shape_real());
    MArray3d rarray(boost::extents[rshape[0]][rshape[1]][rshape[2]]);
    std::vector<int> cshape(distributed_fft3d.get_padded_shape_complex());
    MArray3dc carray(boost::extents[cshape[0]][cshape[1]][cshape[2]]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                rarray[i][j][k] = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }
    distributed_fft3d.transform(rarray, carray);

    auto filename(carray_filename(shape_in));
    write_marray(filename.c_str(), carray);
    fmt::print("wrote {}\n", filename);
}

void
print_on_ranks(Commxx_sptr commxx_sptr, const std::function<void()>& f)
{
    for (int rank = 0; rank < commxx_sptr->get_size(); ++rank) {
        if (rank == commxx_sptr->get_rank()) {
            fmt::print("rank {}: ", rank);
            f();
        }
        commxx_sptr->barrier();
    }
}

void
run_check_distrbuted_fft3d(Shape_t const& shape_in, Shape_t const& cshape_in)
{
    Commxx_sptr commxx_sptr(new Commxx);
    std::vector<int> shape({ shape_in[0], shape_in[1], shape_in[2] });
    Distributed_fft3d distributed_fft3d(shape, commxx_sptr, FFTW_ESTIMATE);
    auto lower = distributed_fft3d.get_lower();
    auto upper = distributed_fft3d.get_upper();
    std::vector<int> rshape(distributed_fft3d.get_padded_shape_real());
    MArray3d rarray(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    MArray3d orig(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    std::vector<int> cshape(distributed_fft3d.get_padded_shape_complex());
    MArray3dc carray(
        boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                orig[i][j][k] = rarray[i][j][k] = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }
    distributed_fft3d.transform(rarray, carray);
    distributed_fft3d.inv_transform(carray, rarray);
    double norm = shape[0] * shape[1] * shape[2];
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                rarray[i][j][k] *= 1.0 / norm;
            }
        }
    }

    // zero out padded region
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < rshape[1]; ++j) {
            for (int k = shape[2]; k < rshape[2]; ++k) {
                rarray[i][j][k] = 0.0;
                orig[i][j][k] = 0.0;
            }
        }
    }

    const double tolerance = 1.0e-10;
    print_on_ranks(commxx_sptr, [&]() {
        fmt::print("Distributed_fft3d roundtrip max error: {}\n",
                   marray_check_equal(rarray, orig, lower, upper, tolerance));
    });

    auto filename(carray_filename(shape_in));
    auto check(read_marray<MArray3dc>(filename.c_str()));
    auto max_error = marray_check_equal(carray, check, lower, upper, tolerance);
    print_on_ranks(commxx_sptr, [&]() {
        fmt::print("Distributed_fft3d carray max error: ({},{})\n",
                   max_error.real(), max_error.imag());
    });
}

std::tuple<double, double>
time_distrbuted_fft3d(Shape_t const& shape_in, Shape_t const& cshape_in)
{
    Commxx_sptr commxx_sptr(new Commxx);
    std::vector<int> shape({ shape_in[0], shape_in[1], shape_in[2] });
    Distributed_fft3d distributed_fft3d(shape, commxx_sptr, FFTW_ESTIMATE);
    auto lower = distributed_fft3d.get_lower();
    auto upper = distributed_fft3d.get_upper();
    std::vector<int> rshape(distributed_fft3d.get_padded_shape_real());
    MArray3d rarray(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    MArray3d orig(
        boost::extents[extent_range(lower, upper)][rshape[1]][rshape[2]]);
    std::vector<int> cshape(distributed_fft3d.get_padded_shape_complex());
    MArray3dc carray(
        boost::extents[extent_range(lower, upper)][cshape[1]][cshape[2]]);
    for (int i = lower; i < upper; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                orig[i][j][k] = rarray[i][j][k] = 1.1 * k + 100 * j + 10000 * i;
            }
        }
    }
    double forward_time =
        run_timing([&]() { distributed_fft3d.transform(rarray, carray); });
    double backward_time =
        run_timing([&]() { distributed_fft3d.inv_transform(carray, rarray); });
    return std::make_tuple(forward_time, backward_time);
}

void
run_check_fftwpp(Shape_t const& shape_in, Shape_t const& cshape_in)
{
    unsigned int nx = shape_in[0];
    unsigned int ny = shape_in[1];
    unsigned int nz = shape_in[2];
    unsigned int nzp = cshape_in[2];

    utils::MPIgroup group(MPI_COMM_WORLD, nx, ny);
    utils::split3 df(nx, ny, nz, group);
    utils::split3 dg(nx, ny, nzp, group, true);

    unsigned int dfZ = df.Z;

    utils::split3 dfgather(nx, ny, dfZ, group);

    Array::array3<Complex> g(dg.x, dg.y, dg.Z, utils::ComplexAlign(dg.n));
    Array::array3<double> f, orig;
    f.Dimension(df.x, df.y, df.Z, utils::doubleAlign(df.n));
    orig.Dimension(df.x, df.y, df.Z, utils::doubleAlign(df.n));

    int divisor = 0;   // Test for best block divisor
    int alltoall = -1; // Test for best alltoall routine
    fftwpp::rcfft3dMPI rcfft(df, dg, f, g,
                             utils::mpiOptions(divisor, alltoall));
    for (int i = 0; i < df.x; ++i) {
        unsigned int ii = df.x0 + i;
        for (int j = 0; j < df.y; ++j) {
            unsigned int jj = df.y0 + j;
            for (int k = 0; k < df.z; ++k) {
                orig(i, j, k) = f(i, j, k) = 1.1 * k + 100 * jj + 10000 * ii;
            }
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    rcfft.Forward(f, g);
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    Commxx_sptr commxx_sptr(new Commxx);
//    print_on_ranks(commxx_sptr,
//                   [&]() { fmt::print("mpifftwpp time = {}\n", time); });

    auto filename(carray_filename(shape_in));
    auto check(read_eigen<Complex>(filename.c_str()));
    Lshape_t clower{ dg.x0, dg.y0, 0 };
    Lshape_t cupper{ dg.x, dg.y, cshape_in[2] };
    write_array3("wtf2.dat", g);
    const double tolerance = 1.0e-10;
    auto max_error =
        general_subcarray_check_equal(clower, cupper, g, check, tolerance);
    print_on_ranks(commxx_sptr, [&]() {
        fmt::print("Distributed_fft3d carray max error: ({},{})\n",
                   max_error.real(), max_error.imag());
    });
    write_array3("wtf.dat", g);
    start = std::chrono::high_resolution_clock::now();
    rcfft.Backward(g, f);
    end = std::chrono::high_resolution_clock::now();
    time =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
//    print_on_ranks(commxx_sptr, [&]() {
//        fmt::print("mpifftwpp backward time = {}\n", time);
//    });

    rcfft.Normalize(f);
    Shape_t lower = { static_cast<int>(df.x0), static_cast<int>(df.y0), 0 };
    Shape_t upper{ static_cast<int>(df.x0 + df.x),
                   static_cast<int>(df.y0 + df.y), static_cast<int>(df.z) };
    auto max = general_subarray_check_equal(lower, upper, f, orig, tolerance);
    print_on_ranks(commxx_sptr,
                   [&]() { fmt::print("mpifftwpp max error = {}\n", max); });
}

std::tuple<double, double>
time_fftwpp(Shape_t const& shape_in, Shape_t const& cshape_in)
{
    unsigned int nx = shape_in[0];
    unsigned int ny = shape_in[1];
    unsigned int nz = shape_in[2];
    unsigned int nzp = cshape_in[2];

    utils::MPIgroup group(MPI_COMM_WORLD, nx, ny);
    utils::split3 df(nx, ny, nz, group);
    utils::split3 dg(nx, ny, nzp, group, true);

    unsigned int dfZ = df.Z;

    utils::split3 dfgather(nx, ny, dfZ, group);

    Array::array3<Complex> g(dg.x, dg.y, dg.Z, utils::ComplexAlign(dg.n));
    Array::array3<double> f, orig;
    f.Dimension(df.x, df.y, df.Z, utils::doubleAlign(df.n));
    orig.Dimension(df.x, df.y, df.Z, utils::doubleAlign(df.n));

    int divisor = 0;   // Test for best block divisor
    int alltoall = -1; // Test for best alltoall routine
    fftwpp::rcfft3dMPI rcfft(df, dg, f, g,
                             utils::mpiOptions(divisor, alltoall));
    for (int i = 0; i < df.x; ++i) {
        unsigned int ii = df.x0 + i;
        for (int j = 0; j < df.y; ++j) {
            unsigned int jj = df.y0 + j;
            for (int k = 0; k < df.z; ++k) {
                orig(i, j, k) = f(i, j, k) = 1.1 * k + 100 * jj + 10000 * ii;
            }
        }
    }
    double forward_time = run_timing([&]() { rcfft.Forward(f, g); });
    double backward_time = run_timing([&]() { rcfft.Backward(g, f); });
    return std::make_tuple(forward_time, backward_time);
}

void
run()
{
    int nx = 32, ny = 32, nz = 256;
    //    int nx = 2, ny = 4, nz = 8;
    int nz_complex = nz / 2 + 1;
    //    unsigned int nz_padded = 2* nz_complex;
    const Shape_t shape{ nx, ny, nz };
    const Shape_t cshape{ nx, ny, nz_complex };

    Commxx commxx;
    if (commxx.get_size() == 1) {
        write_check(shape, cshape);
    }
    run_check_distrbuted_fft3d(shape, cshape);
    show_best_time("Distributed_fft3d",
                   [&]() { return time_distrbuted_fft3d(shape, cshape); });
    run_check_fftwpp(shape, cshape);
    show_best_time("fftwpp",
                   [&]() { return time_distrbuted_fft3d(shape, cshape); });
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    run();
#if 0
    for (int i = 0; i < 3; ++i) {
        run();
    }
    for (int i = 0; i < 3; ++i) {
        run_fftwpp();
    }
    for (int i = 0; i < 3; ++i) {
        run_fftwpp_eigen();
    }
#endif
    MPI_Finalize();
    return 0;
}
