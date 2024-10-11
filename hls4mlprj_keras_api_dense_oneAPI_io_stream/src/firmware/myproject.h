#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"
#include <sycl/ext/intel/prototype/pipes_ext.hpp>
// This file defines the interface to the kernel

using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));
using InPipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<16>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready
));


using InputBeatT = sycl::ext::intel::experimental::StreamingBeat<
    input_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    true>;       // to enable the empty signal


// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
class DenseInputPipeID;
using DenseInputPipe = sycl::ext::intel::experimental::pipe<DenseInputPipeID, InputBeatT, 32, InPipePropertiesT>;
// // hls-fpga-machine-learning insert outputs
// class Layer4OutPipeID;
// using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, result_t, 32, PipeProps>;

// The inter-task pipes need to be declared in the global scope
// hls-fpga-machine-learning insert inter-task pipes
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, layer2_t, 1>;


class MyprojectID;

struct Myproject {

    // kernel property method to config invocation interface
    auto get(sycl::ext::oneapi::experimental::properties_tag) {
        return sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::streaming_interface<>,
                                                           sycl::ext::intel::experimental::pipelined<>};
    }

    SYCL_EXTERNAL void operator()() const;
};

#endif
