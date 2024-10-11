#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
// class Layer2OutPipeID;
// using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, layer2_t, 1>;

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::dense_resource_stream<DenseInputPipe, Layer2OutPipe, config2>> Dense;
    // task_sequence<nnet::elu_stream<Layer2OutPipe, Layer4OutPipe, elu_config4>> Activation;

    // hls-fpga-machine-learning insert layers

    Dense.async(w2, b2);
    // Activation.async(1.0);

    // hls-fpga-machine-learning return
}
