#ifndef REFERENCE_PARTICLE_H_
#define REFERENCE_PARTICLE_H_

#include "four_momentum.h"

class Reference_particle
{
private:
    int charge;
    Four_momentum four_momentum;

public:
    Reference_particle(int charge, double mass, double total_energy)
        : charge(charge)
        , four_momentum(mass, total_energy)
    {
    }

    double get_mass() const { return four_momentum.get_mass(); }

    double get_momentum() const { return four_momentum.get_momentum(); }
};

#endif /* REFERENCE_PARTICLE_H_ */
