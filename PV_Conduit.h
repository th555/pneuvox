#ifndef PV_CONDUIT_H
#define PV_CONDUIT_H

#include <vector>
#include "PV_Chamber.h"
#include "Biquad.h"


// A connection between two pneumatic elements
class PV_Conduit
{
public:
    PV_Conduit(float timeStep, float crossSection, PV_Chamber* from, PV_Chamber* to=nullptr, float extPressure=1.0);

    void update();

    PV_Chamber* from;
    PV_Chamber* to; // This can be NULL, signifying that it is an inlet/outlet connected to
                    // an external pressure
    float open=1.0; // Open=true, it can be traversed
    double conductivity; // A low-pass filtered version of Conduit.open
    float extPressure=1.0; // Only relevant if to=NULL
    double crossSection; // dimensionless, only a multiplier for the flow rate
    float timeStep; // Same as in pneunet

private:
    Biquad filter1;
    Biquad filter2;
    Biquad filter3;
    Biquad filter4;
};



#endif //PV_CONDUIT_H