#include "myproject.h"
#include "parameters.h"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w9.h"
#include "weights/b9.h"

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes

using sycl::ext::intel::experimental::task_sequence;

void Myproject::operator()() const {
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning read in
    [[intel::fpga_register]]
    InputBeatT conv2d_12_input;

    bool keep_going = true;
    
    [[intel::initiation_interval(1)]]
    while (keep_going) {
        conv2d_12_input = Conv2D12InputPipe::read();

        // hls-fpga-machine-learning declare task sequences

        // hls-fpga-machine-learning insert layers

        [[intel::fpga_register]] conv2d_12_result_t layer2_out;
        nnet::conv_2d_cl<input_t, conv2d_12_result_t, config2>(conv2d_12_input.data, layer2_out, w2, b2);
        [[intel::fpga_register]] layer3_t layer3_out;
        nnet::relu<conv2d_12_result_t, layer3_t, relu_config3>(layer2_out, layer3_out);
        [[intel::fpga_register]] layer4_t layer4_out;
        nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out);
        [[intel::fpga_register]] conv2d_13_result_t layer5_out;
        nnet::conv_2d_cl<layer4_t, conv2d_13_result_t, config5>(layer4_out, layer5_out, w5, b5);
        [[intel::fpga_register]] layer6_t layer6_out;
        nnet::relu<conv2d_13_result_t, layer6_t, relu_config6>(layer5_out, layer6_out);
        [[intel::fpga_register]] layer7_t layer7_out;
        nnet::pooling2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out);
        auto& layer8_out = layer7_out;
        [[intel::fpga_register]] dense_6_result_t layer9_out;
        nnet::dense_resource<layer7_t, dense_6_result_t, config9>(layer8_out, layer9_out, w9, b9);
        [[intel::fpga_register]] result_t layer10_out;
        nnet::softmax<dense_6_result_t, result_t, softmax_config10>(layer9_out, layer10_out);

        // hls-fpga-machine-learning return
        Layer10OutPipe::write(layer10_out);

        // stops the kernel when the last input seen.
        keep_going = !conv2d_12_input.eop;
    }
}