#include <iostream>
#include <vector>
#include <math.h>

#include "PV_Controller.h"

#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

PV_Controller::PV_Controller(json data){
    // std::cout << "loading controller...\n";
    json spec = data["controller"];
    if(spec.contains("n_in")){
        n_in = spec["n_in"];
    } else {
        n_in = 2; // sin and cosin
    }
    n_hidden = spec["n_hidden"];
    n_out = spec["n_out"];
    freq = spec["freq"];
    enable_osc = spec.contains("disable_osc") ? !(int)spec["disable_osc"] : 1;

    for(size_t i=0; i<spec["n_hidden"]; i++){
        recurrent_state.push_back(0);
    }

    net = new network<graph>();
    // Building a graph network (with branch and merge) to implement the recurrent (modified) Elman network, according to https://tiny-dnn.readthedocs.io/en/latest/how_tos/How-Tos.html#construct-the-network-model

    /* declare nodes (i.e. layers) */

    // Input from sensors and oscillators
    layers::input& in_sensors = *new layers::input(shape3d(n_in, 1, 1));
    // Input from the context nodes' output at the previous timestep
    layers::input& in_context = *new layers::input(shape3d(n_hidden, 1, 1));
    // Fully connected layers from inputs to hidden
    layers::fc& input_to_hidden = *new layers::fc(n_in, n_hidden);
    layers::fc& context_to_hidden = *new layers::fc(n_hidden, n_hidden);
    // Add the recurrent and normal inputs (after their respective fc layers) before passing them to the hidden layer
    layers::add& hidden_add = *new layers::add(2, n_hidden);
    // Activations of the hidden layer
    tanh_layer& tanh_hidden = *new tanh_layer(n_hidden);
    // FC layer from hidden to output
    layers::fc& output = *new layers::fc(n_hidden, n_out);
    // Activations of the output
    tanh_layer& tanh_output = *new tanh_layer(n_out);
    // element-wise weights from context (previous step) to context layer
    layers::fc& context_to_context = *new layers::fc(n_hidden, n_hidden);
    // element-wise weights from hidden layer to context layer
    layers::fc& hidden_to_context = *new layers::fc(n_hidden, n_hidden);
    // Adding the output of the hidden layer, and the output of the context neurons from the previous step, to feed to the context neurons this step
    layers::add& context_add = *new layers::add(2, n_hidden);
    // Activations of the context layer
    tanh_layer& tanh_context = *new tanh_layer(n_hidden);

    /* connect everything */
    in_sensors << input_to_hidden;
    in_context << context_to_hidden;
    (input_to_hidden, context_to_hidden) << hidden_add << tanh_hidden;
    tanh_hidden << output << tanh_output;
    tanh_hidden << hidden_to_context;
    in_context << context_to_context;
    (hidden_to_context, context_to_context) << context_add << tanh_context;

    /* register to graph */
    construct_graph(*net, (const std::vector<layer *>){&in_sensors, &in_context}, (const std::vector<layer *>){&tanh_output, &tanh_context});

    /* Load the weights and biases */
    vec_t &weights_hidden = input_to_hidden.weights()[0][0];
    vec_t &bias_hidden = input_to_hidden.weights()[1][0];
    for (size_t i=0; i<spec["weights_hidden"].size(); i++){
        weights_hidden[i] = spec["weights_hidden"][i];
    }
    for (size_t i=0; i<spec["bias_hidden"].size(); i++){
        bias_hidden[i] = spec["bias_hidden"][i];
    }

    vec_t &weights_out = output.weights()[0][0];
    vec_t &bias_out = output.weights()[1][0];
    for (size_t i=0; i<spec["weights_out"].size(); i++){
        weights_out[i] = spec["weights_out"][i];
    }
    for (size_t i=0; i<spec["bias_out"].size(); i++){
        bias_out[i] = spec["bias_out"][i];
    }

    vec_t &weights_recurrent = context_to_hidden.weights()[0][0];
    for (size_t i=0; i<spec["weights_recurrent"].size(); i++){
        weights_recurrent[i] = spec["weights_recurrent"][i];
    }

    vec_t &bias_context = context_to_context.weights()[1][0];
    for (size_t i=0; i<spec["bias_context"].size(); i++){
        bias_context[i] = spec["bias_context"][i];
    }

    // Here we only need the weights from node_i to node_i, since we use an fc layer
    // we must set the others to 0.
    vec_t &weights_recurrent_self = context_to_context.weights()[0][0];
    for (size_t i=0; i<weights_recurrent_self.size(); i++){
        weights_recurrent_self[i] = 0;
    }
    for (size_t i=0; i<spec["weights_recurrent_self"].size(); i++){
        weights_recurrent_self[i*n_hidden + i] = spec["weights_recurrent_self"][i];
    }

    // Idem
    vec_t &weights_context = hidden_to_context.weights()[0][0];
    for (size_t i=0; i<weights_context.size(); i++){
        weights_context[i] = 0;
    }
    for (size_t i=0; i<spec["weights_context"].size(); i++){
        weights_context[i*n_hidden + i] = spec["weights_context"][i];
    }
    // Create a visualization in dot language
    // render to a picture like this:
    // dot -Tgif nn_viz.txt -o graph.gif
    // 
    // std::ofstream ofs("nn_viz.txt");
    // graph_visualizer viz(*net, "graph");
    // viz.generate(ofs);

    // std::cout << *net << "\n";
}

std::vector<float> PV_Controller::evaluate(double timestep, std::vector<float> sensors){ //timestep in seconds
    // Update oscs
    phase += 2*M_PI * timestep * freq;
    if(phase >= 2*M_PI){
        phase -= 2*M_PI;
    }
    if(enable_osc){
        /* Construct input from oscs and sensors */
        inputs = {sin(phase), cos(phase)};
    } else {
        inputs = {};
    }
    for(int i=0; i<std::min((int)sensors.size(), 10); i++){ // TODO magic number 10
        inputs.push_back(sensors[i]);
    }
    for(int i=inputs.size(); i<n_in; i++){ // pad with zeroes
        inputs.push_back(0);
    }

    std::vector<float> output;
    vec_t in1(inputs.begin(), inputs.end());
    vec_t in2(recurrent_state.begin(), recurrent_state.end());
    tensor_t net_inputs = {in1, in2};
    tensor_t net_outputs = net->predict(net_inputs);
    recurrent_state = {net_outputs[1].begin(), net_outputs[1].end()};
    output = {net_outputs[0].begin(), net_outputs[0].end()};

    // Scale from (-1, 1) to (0, 1)
    for(int i=0; i<output.size(); i++){
        output[i] += 1.0;
        output[i] /= 2.0;
    }

    // std::cout << "output: ";
    // for (auto const& c : output){std::cout << c << ' ';}std::cout<<"\n";

    return output;
}

