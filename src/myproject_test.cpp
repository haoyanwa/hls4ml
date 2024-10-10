#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "firmware/myproject.h"
#include "firmware/parameters.h"

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/interfaces.hpp>

#include "exception_handler.hpp"
// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

#if not defined(IS_BSP)
using sycl::ext::intel::experimental::property::usm::buffer_location;
#endif

int main(int argc, char **argv)
{

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

    auto device = q.get_device();

    // make sure the device supports USM host allocations
    // if (!device.has(sycl::aspect::usm_host_allocations)) {
    //     std::cerr << "This design must either target a board that supports USM "
    //                  "Host/Shared allocations, or IP Component Authoring. "
    //               << std::endl;
    //     std::terminate();
    // }

    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

    // (haoyanwa) constants.
    // kInputSz:
    // kNumBatch * N_INPUT_1_1 * N_INPUT_2_1
    //     4           32            3
    //   Batch      N_Samples    N_Feature

    // kOutputSz:
    // kNumBatch * N_OUT_4
    //     4         16
    //   Batch    N_Feature

    constexpr size_t kNumBatch = 32;
    constexpr size_t kinputSz = kNumBatch * N_INPUT_1_1 * N_INPUT_2_1;
    constexpr size_t kOutputSz = kNumBatch * N_OUT_4;
    std::cout << "INFO: Using default input with " << kNumBatch << " batch" << std::endl;

    // hls-fpga-machine-learning insert zero
    // (haoyanwa) change the input and output to device ptr.
// #if defined(IS_BSP)
//     float *vals_device_ptr = sycl::malloc_host<float>(kinputSz, q);
//     if (vals_device_ptr == nullptr)
//     {
//         std::cerr << "ERROR: host allocation failed for input\n";
//         return 1;
//     }
//     float *output_device_ptr = sycl::malloc_host<float>(kOutputSz, q);
//     if (output_device_ptr == nullptr)
//     {
//         std::cerr << "ERROR: host allocation failed for output\n";
//         return 1;
//     }
// #else
//     float *vals_device_ptr = sycl::malloc_shared<float>(kinputSz, q, sycl::property_list{buffer_location(kInputBufferLocation)});
//     float *output_device_ptr = sycl::malloc_shared<float>(kOutputSz, q, sycl::property_list{buffer_location(kOutputBufferLocation)});
// #endif
    try
    {
        [[intel::fpga_register]]
        typename nnet::ExtractPipeType<Conv1DInputPipe>::value_type ctype;

        constexpr size_t dstTypeSize = std::tuple_size<input_t>{};

        q.single_task(Myproject{});
        for (size_t i = 0; i < kinputSz / dstTypeSize; i++)
        {
            for (size_t j = 0; j < dstTypeSize; j++)
            {
                ctype.data[j] = 1.0;
            }
            ctype.sop = (i == 0);
            ctype.eop = (i == (kinputSz / dstTypeSize - 1));
            Conv1DInputPipe::write(q, ctype);
        }

        constexpr auto srcTypeSize = std::tuple_size<typename nnet::ExtractPipeType<Layer4OutPipe>::value_type>{};
        double buffer[kOutputSz];
        [[intel::fpga_register]]
        typename nnet::ExtractPipeType<Layer4OutPipe>::value_type octype;

        for (size_t i = 0; i < kOutputSz / srcTypeSize; i++)
        {
            octype = Layer4OutPipe::read(q);
            for (size_t j = 0; j < srcTypeSize; j++)
            {
                buffer[i * srcTypeSize + j] = octype[j].to_double();
            }
        }

        for (int j = 0; j < kOutputSz; j++)
        {
            std::cout << buffer[j] << " ";
        }

        q.wait();
    }
    catch (sycl::exception const &e)
    {
        // Catches exceptions in the host code.
        std::cerr << "Caught a SYCL host exception:\n"
                  << e.what() << "\n";

        // Most likely the runtime couldn't find FPGA hardware!
        if (e.code().value() == CL_DEVICE_NOT_FOUND)
        {
            std::cerr << "If you are targeting an FPGA, please ensure that your "
                         "system has a correctly configured FPGA board.\n";
            std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
            std::cerr << "If you are targeting the FPGA emulator, compile with "
                         "-DFPGA_EMULATOR.\n";
        }
        std::terminate();
    }

    std::cout << std::endl;

    std::cout << "Done." << std::endl;

    return 0;
}
