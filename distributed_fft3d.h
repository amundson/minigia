#ifndef DISTRIBUTED_FFT3D_H_
#define DISTRIBUTED_FFT3D_H_

#include <fftw3.h>
#include <fftw3-mpi.h>
#include <vector>
#include <string>
#include "boost/shared_ptr.hpp"
#include "multi_array_typedefs.h"
#include "commxx.h"

class Distributed_fft3d
{
private:
    fftw_plan plan, inv_plan;
    double *data;
    fftw_complex *workspace;
    int lower, upper;
    std::vector<int> uppers, lengths;
    int local_size_real, local_size_complex;
    std::vector<int > shape;
    bool have_local_data;
    Commxx_sptr comm_sptr;
    void
    calculate_uppers_lengths();
public:
    Distributed_fft3d(std::vector<int > const& shape, Commxx_sptr comm_sptr,
            int planner_flags = FFTW_ESTIMATE,
            std::string const& wisdom_filename = "");
    Commxx_sptr
    get_comm_sptr();
    Commxx &
    get_comm();
    int
    get_lower() const;
    int
    get_upper() const;
    std::vector<int > const&
    get_uppers();
    std::vector<int > const&
    get_lengths();
    std::vector<int > const&
    get_shape() const;
    std::vector<int >
    get_padded_shape_real() const;
    std::vector<int >
    get_padded_shape_complex() const;
    void
    transform(MArray3d_ref & in, MArray3dc_ref & out);
    void
    inv_transform(MArray3dc_ref & in, MArray3d_ref & out);
    double
    get_roundtrip_normalization() const;
    ~Distributed_fft3d();
};

typedef boost::shared_ptr<Distributed_fft3d > Distributed_fft3d_sptr; // syndoc:include
#endif /* DISTRIBUTED_FFT3D_H_ */
