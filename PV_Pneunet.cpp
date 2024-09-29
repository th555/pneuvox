#include "PV_Pneunet.h"

PV_Pneunet::PV_Pneunet(float timeStep) : timeStep(timeStep) {
}

void PV_Pneunet::addChamber(PV_Chamber* chamber){
    chambers.push_back(chamber);
}

void PV_Pneunet::addValve(PV_Valve* valve){
    valves.push_back(valve);
}

PV_Conduit* PV_Pneunet::connect(PV_Chamber* from, PV_Chamber* to, float crossSection){
    PV_Conduit* conduit = new PV_Conduit(timeStep, crossSection, from, to);
    conduits.push_back(conduit);

    return conduit;
}

PV_Conduit* PV_Pneunet::connectExternal(PV_Chamber* chamber, float crossSection, float extPressure){
    PV_Conduit* conduit = new PV_Conduit(timeStep, crossSection, chamber, nullptr, extPressure);
    conduits.push_back(conduit);

    return conduit;
}



float PV_Pneunet::pressure(){
    return quantity()/volume();
}

float PV_Pneunet::volume(){
    float volume = 0;
    for(std::size_t i=0; i<chambers.size(); i++){
        volume += chambers[i]->volume;
    }
    return volume;
}

float PV_Pneunet::quantity(){
    float quantity = 0;
    for(std::size_t i=0; i<chambers.size(); i++){
        quantity += chambers[i]->quantity;
    }
    return quantity;
}

/*
Update sequence is as follows:
[x] clear external forces on voxels (in child.preUpdate)
[x] update volumes based on current state of the voxels (in child.preUpdate -> child.updateVolume)

{simultaneously
[x] update manual valves based on external signals
[x] update voxel-strain-sensitive valves based on current state of the voxels
[x] update pneumatic pressure-sensitive valves based on current state of the pressure
}

[x] for each conduit: transfer quantities of air between the connected chambers (or inlet/outlet) based on the pressure difference
[x] update pressure (and apply external forces on voxels due to pressure)

[x] voxels move according to the mass/spring model including external forces
*/
void PV_Pneunet::update(){
    for(std::size_t i=0; i<chambers.size(); i++){
        chambers[i]->preUpdate();
    }
    for(std::size_t i=0; i<valves.size(); i++){
        valves[i]->preUpdate();
    }
    for(std::size_t i=0; i<chambers.size(); i++){
        chambers[i]->update();
    }

    for(std::size_t i=0; i<conduits.size(); i++){
        // Conduit->update(), conduit->open() etc..
        PV_Conduit* c = conduits[i];
        c->update();
        float p1 = c->from->pressure;
        float p2;
        if(c->to){
            p2 = c->to->pressure;
        } else {
            p2 = c->extPressure;
        }
        float qty = c->conductivity * (p1 - p2) * timeStep; // Quantity to transfer from "from" to "to" (can be negative for other way around)
        c->from->quantity -= qty;
        if(c->to){
            c->to->quantity += qty;
        }
    }
}
