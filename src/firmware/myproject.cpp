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

    // hls-fpga-machine-learning read in

    // (haoyanwa) Test restartable kernel
    bool keep_going = true;

    [[intel::fpga_register]] InputBeatT conv1d_input_beat;
    [[intel::fpga_register]] conv1d_result_t layer2_out;
    [[intel::fpga_register]] layer3_t layer3_out;
    [[intel::fpga_register]] result_t layer4_out;

    [[intel::initiation_interval(1)]]
    while (keep_going) {
        conv1d_input_beat = Conv1DInputPipe::read();

        auto conv1d_input = conv1d_input_beat.data;

        // hls-fpga-machine-learning declare task sequences
        nnet::conv_1d_cl<input_t, conv1d_result_t, config2>(conv1d_input, layer2_out, w2, b2);
        nnet::relu<conv1d_result_t, layer3_t, relu_config3>(layer2_out, layer3_out);
        nnet::gru<layer3_t, result_t, config4>(layer3_out, layer4_out, w4, wr4, b4, br4);

        // hls-fpga-machine-learning return
        Layer4OutPipe::write(layer4_out);

        // stops the kernel when the last input seen.
        keep_going = !conv1d_input_beat.eop;
    }

}
