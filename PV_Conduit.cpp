#include "PV_Conduit.h"
#include <iostream>

PV_Conduit::PV_Conduit(float timeStep, float crossSection, PV_Chamber* from, PV_Chamber* to, float extPressure){
    this->timeStep = timeStep;
    this->crossSection = crossSection;
    this->from = from;
    this->to = to;
    this->extPressure = extPressure;

    // Set LPF based on timestep (see https://www.earlevel.com/main/2012/11/26/biquad-c-source-code/ and https://www.earlevel.com/main/2021/09/02/biquad-calculator-v3/)
    double flt_freq = 10.0; // cutoff freq in Hz
    // Use 4 filters in series for steeper cutoff (48 db/oct instead of 12)
    filter1.setBiquad(bq_type_lowpass, timeStep * flt_freq, 0.707, 0);
    filter2.setBiquad(bq_type_lowpass, timeStep * flt_freq, 0.707, 0);
    filter3.setBiquad(bq_type_lowpass, timeStep * flt_freq, 0.707, 0);
    filter4.setBiquad(bq_type_lowpass, timeStep * flt_freq, 0.707, 0);

}

void PV_Conduit::update(){
    /* It might get negative or >1 due to filter instability/resonance so we clip */
    double openness = filter1.process(
        filter2.process(
            filter3.process(
                filter4.process(open))));
    openness = openness < 0 ? 0 : openness;
    openness = openness > 1 ? 1 : openness;
    conductivity = openness * crossSection;
}
