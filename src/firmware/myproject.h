#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

// (haoyanwa) Debug
#if FPGA_EMULATOR
#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

#define PRINTF(format, ...)                                    \
  {                                                            \
    static const CL_CONSTANT char _format[] = format;          \
    sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }
#endif

// This file defines the interface to the kernel

// currently this is fixed
using PipeProps = decltype(sycl::ext::oneapi::experimental::properties(sycl::ext::intel::experimental::ready_latency<0>));

// Need to declare the input and output pipes

// hls-fpga-machine-learning insert inputs
class Conv1DInputPipeID;
using Conv1DInputPipe = sycl::ext::intel::experimental::pipe<Conv1DInputPipeID, input_t, 0, PipeProps>;
// hls-fpga-machine-learning insert outputs
class Layer4OutPipeID;
using Layer4OutPipe = sycl::ext::intel::experimental::pipe<Layer4OutPipeID, result_t, 0, PipeProps>;

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
