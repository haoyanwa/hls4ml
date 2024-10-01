#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/wr4.h"
#include "weights/b4.h"
#include "weights/br4.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************


    auto conv1d_input = Conv1DInputPipe::read();

    // hls-fpga-machine-learning declare task sequences

    // hls-fpga-machine-learning insert layers

    [[intel::fpga_register]] conv1d_result_t layer2_out;
    nnet::conv_1d_cl<input_t, conv1d_result_t, config2>(conv1d_input, layer2_out, w2, b2);
    [[intel::fpga_register]] layer3_t layer3_out;
    nnet::relu<conv1d_result_t, layer3_t, relu_config3>(layer2_out, layer3_out);
    [[intel::fpga_register]] result_t layer4_out;
    nnet::gru<layer3_t, result_t, config4>(layer3_out, layer4_out, w4, wr4, b4, br4);

    // hls-fpga-machine-learning return
    Layer4OutPipe::write(layer4_out);
}
