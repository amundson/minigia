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

    void set_four_momentum(Four_momentum const& four_momentum) { this->four_momentum = four_momentum; }

    double get_charge() const { return charge; }

    double get_mass() const { return four_momentum.get_mass(); }

    double get_momentum() const { return four_momentum.get_momentum(); }

    double get_gamma() const { return four_momentum.get_gamma(); }
};

#endif /* REFERENCE_PARTICLE_H_ */
