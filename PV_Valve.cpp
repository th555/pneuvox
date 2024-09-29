#include "PV_Valve.h"
#include "PV_Conduit.h"
#include <iostream>


PV_Valve::PV_Valve(PV_Conduit* conduits_[], int n, bool inverts[]){
    for(int i=0; i<n; i++){
        bool invert = inverts ? inverts[i] : 0;
        PV_Conduit* conduit = conduits_[i];

        // correctly initialize all conduits
        conduit->open = !invert;
        conduits.push_back(conduit);
        inversions.push_back(invert);
    }
}


void PV_Valve::preUpdate(){
    float newState = getState();
    if(newState != switchState){
        switchState = newState;
        for(std::size_t i=0; i<conduits.size(); i++){
            conduits[i]->open = switchState * (inversions[i] ? -1.0 : 1.0);
        }
    }
}


PV_ValveToggle::PV_ValveToggle(PV_Conduit* conduits_[], int n, bool inverts[]) : PV_Valve(conduits_, n, inverts)
{};

float PV_ValveToggle::getState(){
    return state;
}

void PV_ValveToggle::set(float open){
    state = open;
}

