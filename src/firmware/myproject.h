#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "defines.h"

#include <sycl/ext/intel/prototype/pipes_ext.hpp>

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


using InputBeatT = sycl::ext::intel::experimental::StreamingBeat<
    input_t,     // type carried over this Avalon streaming interface's data signal.
    true,         // enable startofpacket and endofpacket signals
    false>;       // to disable the empty signal

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

class StopPipeID;
// StopPipe is mapped to the CSR. This way a kernel can be controlled by a
// memory-mapped host while it executes.
using CsrPipeProperties = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::protocol<
        // Write-only, so no no need for protocol_name::avalon_mm_uses_ready
        sycl::ext::intel::experimental::protocol_name::avalon_mm_uses_ready>));
using StopPipe = sycl::ext::intel::experimental::pipe<StopPipeID, bool, 0, CsrPipeProperties>;

class IDInputDMA;
class IDOutputDMA;

template <class srcType, class dest_pipe, size_t SIZE> 
struct DMA_convert_data {
#if !defined(IS_BSP)
    // Customizing mmhost only supported when targetting an FPGA part/family
    sycl::ext::oneapi::experimental::annotated_arg<srcType *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::dwidth<16>,
          sycl::ext::intel::experimental::buffer_location<1>,
          sycl::ext::intel::experimental::read_write_mode_read})>
#else
    // Declare USM for data input.
    srcType *const
#endif
        src;

    [[intel::kernel_args_restrict]]
    void operator()() const {
        
#if defined(IS_BSP)
        // When targeting a BSP, we instruct the compiler that this pointer lives on
        // the device.
        // Knowing this, the compiler won't generate hardware to potentially get
        // data from the host.
        sycl::ext::intel::device_ptr<srcType> src_ptr(src);
#else
        // Device pointers are not supported when targeting an FPGA family/part
        srcType *src_ptr(src);
#endif

        constexpr auto dstTypeSize = std::tuple_size<typename nnet::ExtractPipeType<dest_pipe>::value_type>{};
        for (size_t i = 0; i < SIZE / dstTypeSize; i++) {
            typename nnet::ExtractPipeType<dest_pipe>::value_type ctype;
            for (size_t j = 0; j < dstTypeSize; j++) {
                ctype[j] = src_ptr[i * dstTypeSize + j];
                // ctype.data[j] = src_ptr[i * dstTypeSize + j];
            }
            // ctype.sop = (i == 0);
            // ctype.eop = (i == (SIZE / dstTypeSize - 1));
            dest_pipe::write(ctype);
        }
        PRINTF("InputDMA Sent data\n");
    }
};

template <class src_pipe, class dstType, size_t SIZE> 
struct DMA_convert_data_back {
#if !defined(IS_BSP)
    // Customizing mmhost only supported when targetting an FPGA part/family
    sycl::ext::oneapi::experimental::annotated_arg<dstType *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::dwidth<16>,
          sycl::ext::intel::experimental::buffer_location<2>,
          sycl::ext::intel::experimental::read_write_mode_write})>
#else
    // Declare USM for data input.
    dstType *const
#endif
        dst;

    [[intel::kernel_args_restrict]]
    void operator()() const {
#if defined(IS_BSP)
        // When targeting a BSP, we instruct the compiler that this pointer lives on
        // the device.
        // Knowing this, the compiler won't generate hardware to potentially get
        // data from the host.
        sycl::ext::intel::device_ptr<dstType> dst_ptr(dst);
#else
        // Device pointers are not supported when targeting an FPGA family/part
        dstType *dst_ptr(dst);
#endif
        constexpr auto srcTypeSize = std::tuple_size<typename nnet::ExtractPipeType<src_pipe>::value_type>{};
        for (size_t i = 0; i < SIZE / srcTypeSize; i++) {
            auto ctype = src_pipe::read();
            PRINTF("Output DMA received data\n");
            for (size_t j = 0; j < srcTypeSize; j++) {
                dst_ptr[i * srcTypeSize + j] = ctype[j].to_double();
            }
        }
        PRINTF("Output DMA wrote output to device_ptr\n");
    }
};

#endif
