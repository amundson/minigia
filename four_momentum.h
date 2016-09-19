#ifndef FOUR_MOMENTUM_H_
#define FOUR_MOMENTUM_H_

#include <cmath>
#include <stdexcept>

class Four_momentum
{
private:
    double mass, energy, momentum, gamma, beta;
    void update_from_gamma()
    {
        if (gamma < 1.0) {
            throw std::range_error("Four_momentum: gamma not >= 1.0");
        }
        energy = gamma * mass;
        beta = sqrt(1.0 - 1.0 / (gamma * gamma));
        momentum = gamma * beta * mass;
    }

public:
    Four_momentum(double mass, double total_energy)
        : mass(mass)
    {
        set_total_energy(total_energy);
    }

    void set_total_energy(double total_energy)
    {
        gamma = total_energy / mass;
        update_from_gamma();
    }

    double get_mass() const { return mass; }

    double get_momentum() const { return momentum; }
};

#endif /* FOUR_MOMENTUM_H_ */
