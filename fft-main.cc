#include <chrono>
#include <fstream>
#include <iostream>

#include "compare.h"
#include "distributed_fft3d.h"

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
run()
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
              << marray_check_equal(carray, check, tolerance) << std::endl;
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
              << marray_check_equal(rarray, orig, tolerance) << std::endl;
}

int
main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    for (int i = 0; i < 10; ++i) {
        run();
    }
    MPI_Finalize();
    return 0;
}
