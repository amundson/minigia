#ifndef COLLECTIVE_OPERATOR_H
#define COLLECTIVE_OPERATOR_H

#include "bunch.h"

class Collective_operator
{
public:
    virtual void apply(Bunch& bunch, double time_step, int verbosity) = 0;
    virtual ~Collective_operator() {}
};

#endif // COLLECTIVE_OPERATOR_H
