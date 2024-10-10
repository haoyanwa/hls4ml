#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
class Conv2D12InputPipeID;
using Conv2D12InputPipe = sycl::ext::intel::experimental::pipe<Conv2D12InputPipeID, input_t, 0, PipeProps>;
// hls-fpga-machine-learning insert outputs
class Layer10OutPipeID;
using Layer10OutPipe = sycl::ext::intel::experimental::pipe<Layer10OutPipeID, result_t, 0, PipeProps>;

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
