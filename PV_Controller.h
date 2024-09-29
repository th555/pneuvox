#ifndef PV_CONTROLLER_H
#define PV_CONTROLLER_H

#include "json.hpp"

using json = nlohmann::json;

// We can't include tiny_dnn here because of a conflict in the #definition of PI...
// Therefore we must do some forward declarations to be able to use tinydnn stuff
// as part of our class.
namespace tiny_dnn{
    class sequential;
    class graph;
    template <typename NetType>
    class network;
};
using namespace tiny_dnn;

class PV_Controller
{
public:
    PV_Controller(json data);
    std::vector<float> evaluate(double timestep, std::vector<float> sensors = {});
    std::vector<float> inputs;
    int n_in;
    int n_recurrent;
    int n_hidden;
    int n_out;
    bool enable_osc = true;

private:
    network<graph>* net; // tiny-dnn network


    std::vector<float> recurrent_state;

    double phase = 0; // current phase of the oscillator, between 0 and 2pi
    double freq; // oscillator frequency in hz
};













#endif //PV_CONTROLLER_H
