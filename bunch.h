#ifndef BUNCH_H_
#define BUNCH_H_

class Bunch {
   public:
    static const int x = 0;
    static const int xp = 1;
    static const int y = 2;
    static const int yp = 3;
    static const int z = 4;
    static const int zp = 5;
    static const int cdt = 4;
    static const int dpop = 5;
    static const int id = 6;
    static const int particle_size = 7;

    struct AView {
        double * RESTRICT x;
        double * RESTRICT xp;
        double * RESTRICT y;
        double * RESTRICT yp;
        double * RESTRICT cdt;
        double * RESTRICT dpop;
    };

   private:
    Reference_particle reference_particle;
    double* storage;
    MArray2d_ref* local_particles;
    int local_num;
    AView aview;

   public:
    Bunch(int total_num, int mpi_size, int mpi_rank) : reference_particle() {
        local_num = total_num / mpi_size;
        storage =
#ifdef MM_MALLOC
              (double*)_mm_malloc(local_num * particle_size * sizeof(double), 64);
#else
              (double *)malloc(local_num * particle_size * sizeof(double));
#endif
        local_particles = new MArray2d_ref(
            storage, boost::extents[local_num][Bunch::particle_size],
                         boost::fortran_storage_order());
        double *origin = local_particles->origin();
        aview.x = origin + local_num*Bunch::x;
        aview.xp = origin + local_num*Bunch::xp;
        aview.y = origin + local_num*Bunch::y;
        aview.yp = origin + local_num*Bunch::yp;
        aview.cdt = origin + local_num*Bunch::cdt;
        aview.dpop = origin + local_num*Bunch::dpop;
        for (int part = 0; part < local_num; ++part) {
            int index = part + mpi_rank * mpi_size;
            (*local_particles)[part][Bunch::x] = 1.0e-6 * index;
            (*local_particles)[part][Bunch::xp] = 1.1e-8 * index;
            (*local_particles)[part][Bunch::y] = 1.3e-6 * index;
            (*local_particles)[part][Bunch::yp] = 1.4e-8 * index;
            (*local_particles)[part][Bunch::z] = 1.5e-4 * index;
            (*local_particles)[part][Bunch::zp] = 1.5e-7 * index;
            (*local_particles)[part][Bunch::id] = index;
        }
    }

    Reference_particle const& get_reference_particle() const {
        return reference_particle;
    }

    MArray2d_ref get_local_particles() { return *local_particles; }

    double get_mass() const { return dummy_mass; }

    int get_local_num() const { return local_num; }

    void set_arrays(double * RESTRICT &xa, double * RESTRICT &xpa,
                    double * RESTRICT &ya, double * RESTRICT &ypa,
                    double * RESTRICT &cdta, double * RESTRICT &dpopa)
    {
        double *origin = local_particles->origin();
        xa = origin + local_num*Bunch::x;
        xpa = origin + local_num*Bunch::xp;
        ya = origin + local_num*Bunch::y;
        ypa = origin + local_num*Bunch::yp;
        cdta = origin + local_num*Bunch::cdt;
        dpopa = origin + local_num*Bunch::dpop;
    }

    AView get_aview() { return aview; }

    virtual ~Bunch() {
        delete local_particles;
#ifdef MM_MALLOC
        _mm_free(storage);
#else
        free(storage);
#endif
    }
};

#endif /* BUNCH_H_ */
