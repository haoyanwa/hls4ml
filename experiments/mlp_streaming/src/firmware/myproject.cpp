#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, fc1_result_t, 1>;
class Layer4OutPipeID;
using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, layer4_t, 1>;
class Layer5OutPipeID;
using Layer5OutPipe = sycl::ext::intel::experimental::pipe<Layer5OutPipeID, fc2_result_t, 1>;

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in

    // hls-fpga-machine-learning declare task sequences
    task_sequence<nnet::dense_resource_stream<Fc1InputPipe, Layer2OutPipe, config2>> fc1;
    task_sequence<nnet::relu_stream<Layer2OutPipe, Layer4OutPipe, relu_config4>> relu1;
    task_sequence<nnet::dense_resource_stream<Layer4OutPipe, Layer5OutPipe, config5>> fc2;
    task_sequence<nnet::relu_stream<Layer5OutPipe, Layer7OutPipe, relu_config7>> relu2;

    // hls-fpga-machine-learning insert layers

    fc1.async(w2, b2);
    relu1.async();
    fc2.async(w5, b5);
    relu2.async();

    // hls-fpga-machine-learning return
}
