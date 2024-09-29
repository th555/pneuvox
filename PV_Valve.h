#ifndef PV_VALVE_H
#define PV_VALVE_H

#include "PV_Conduit.h"
#include "PV_Chamber.h"
#include <vector>
#include "VX_Voxel.h"
#include "VX_Link.h"



// Base class for valves. Subclasses can differ in the way the state changes (manually, based on pressure, voxel strain etc...)
class PV_Valve
{
public:
    PV_Valve(PV_Conduit* conduits_[], int n=1, bool inverts[]=nullptr); /* Create a valve with n independent port pairs, use the inverts list
    to create normally-open (default, [false]), normally-closed [true] or bistable (n=2, [false, true]) valves */

    void preUpdate();

    virtual float getState() = 0; // 0: closed, 1: open

    std::vector<PV_Conduit*> conduits; // The conduits between the valve's (controlled) port pairs
    // The conduits can also be used as an alternative way to access the ports,
    // that might be more logical when a valve has multiple ports, or when one has to differentiate
    // between the input and output port pairs of a Pressure valve.

private:
    float switchState = 1.0;
    std::vector<bool> inversions;
};



// Simple switch that can be flipped by calling set()
class PV_ValveToggle : public PV_Valve
{
public:
    PV_ValveToggle(PV_Conduit* conduits_[], int n=1, bool inverts[]=nullptr);
    float getState();

    void set(float open);
private:
    float state = 1.0;
};


#endif //PV_VALVE_H