#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

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

using OutputBeatT = sycl::ext::intel::experimental::StreamingBeat<
    result_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    true>;       // to enable the empty signal


// hls-fpga-machine-learning insert inputs
class Fc1InputPipeID;
using Fc1InputPipe = sycl::ext::intel::experimental::pipe<Fc1InputPipeID, InputBeatT, 32, InPipePropertiesT>;
// hls-fpga-machine-learning insert outputs
class Layer7OutPipeID;
using Layer7OutPipe = sycl::ext::intel::experimental::pipe<Layer7OutPipeID, OutputBeatT, 32, PipeProps>;


using l2b_t = sycl::ext::intel::experimental::StreamingBeat<
    fc1_result_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    true>;       // to enable the empty signal
class Layer2OutPipeID;
using Layer2OutPipe = sycl::ext::intel::experimental::pipe<Layer2OutPipeID, l2b_t, 32>;


using l4b_t = sycl::ext::intel::experimental::StreamingBeat<
    layer4_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    true>;       // to enable the empty signal
class Layer4OutPipeID;
using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, l4b_t, 32>;


using l5b_t = sycl::ext::intel::experimental::StreamingBeat<
    fc2_result_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    true>;       // to enable the empty signal
class Layer5OutPipeID;
using Layer5OutPipe = sycl::ext::intel::experimental::pipe<Layer5OutPipeID, l5b_t, 32>;


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
