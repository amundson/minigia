#ifndef PHYSICAL_CONSTANTS_H_
#define PHYSICAL_CONSTANTS_H_
#include "math_constants.h"
#include <string>

#ifndef PDG_VERSION
#define PDG_VERSION 2012
#endif

namespace pconstants
{
    const double proton_mass = 0.938272046; // Mass of proton [GeV/c^2]
    const double e = 1.602176565e-19; // Charge of proton [C]
    const double c = 299792458.0; // Speed of light [m/s]

    const double mu0 = 4*mconstants::pi*1.0e-7; // Permittivity of free space [H/m]
    const double epsilon0 = 1.0/(c*c*mu0); // Permeability of free space [F/m]

    const int proton_charge = 1; // Charge in units of e
}

#endif /* PHYSICAL_CONSTANTS_H_ */
