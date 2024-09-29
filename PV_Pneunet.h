#ifndef PV_PNEUNET_H
#define PV_PNEUNET_H

#include "PV_Conduit.h"
#include "PV_Chamber.h"
#include "PV_Valve.h"
#include <vector>


class PV_Pneunet
{
public:
    PV_Pneunet(float timeStep); // constructor

    void addChamber(PV_Chamber* chamber);
    /* Connect two pneumatic elements by adding a conduit between them
    They must be add()ed separately! */
    PV_Conduit* connect(PV_Chamber* from, PV_Chamber* to, float crossSection); // connect two chambers
    PV_Conduit* connectExternal(PV_Chamber* chamber, float crossSection, float extPressure=1.0); // connect one chamber to external pressure (ambient or pressurized)
    void addValve(PV_Valve* valve);


    void update();

    // Gather various quantities from all children
    float pressure(); // Pressure of all connected children, i.e. sum(quantity)/sum(volume)
    float volume();
    float quantity();

    float timeStep; // In seconds, should be set to the Vx timestep on creation
    std::vector<PV_Chamber*> chambers; // All pneumatic elements managed by this pneunet

private:
    std::vector<PV_Valve*> valves; // The valves which control the state of the conduits (not a parallel array)
    std::vector<PV_Conduit*> conduits; // All connections between the elements
};





#endif // PV_PNEUNET_H
