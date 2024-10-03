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


int main(int argc, char **argv) {

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

    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

    std::string RESULTS_LOG = "tb_data/results.log";
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;

    const unsigned int num_iterations = 2;
    std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                << " invocations." << std::endl;

    // (haoyanwa) constants.
    // kInputSz:
    // kNumBatch * N_INPUT_1_1 * N_INPUT_2_1
    //     4           32            3
    //   Batch      N_Samples    N_Feature

    // kOutputSz:
    // kNumBatch * N_OUT_4
    //     4         16
    //   Batch    N_Feature

    constexpr size_t kNumBatch = 4;
    constexpr size_t kinputSz = kNumBatch * N_INPUT_1_1 * N_INPUT_2_1;
    constexpr size_t kOutputSz = kNumBatch * N_OUT_4;

    // hls-fpga-machine-learning insert zero
    // (haoyanwa) change the input and output to device ptr.
#if defined(IS_DSP)
    float *vals_device_ptr = sycl::malloc_device<float>(kinputSz, q);
    float *output_device_ptr = sycl::malloc_device<float>(kOutputSz, q);
#else
    float *vals_device_ptr = sycl::malloc_shared<float>(kinputSz, q, sycl::property_list{buffer_location(kInputBufferLocation)});
    float *output_device_ptr = sycl::malloc_shared<float>(kOutputSz, q, sycl::property_list{buffer_location(kOutputBufferLocation)});
#endif

    // (haoyanwa) Defines the input buffer and output buffer.
    float vals[kinputSz]; 
    float outputs[kOutputSz];

    // Init to all 1s.
    for (int j = 0 ; j < kinputSz; j++) {
        vals[j] = 1.0; 
    }

    // hls-fpga-machine-learning insert top-level-function
    for (int i = 0; i < num_iterations; i++) {
        // copy the input data to the device memory and wait for the copy to
        // finish
        q.memcpy(vals_device_ptr, vals, kinputSz * sizeof(float)).wait();
        
        // nnet::convert_data<float, Conv1DInputPipe, N_INPUT_1_1*N_INPUT_2_1>(q, vals);
        // (haoyanwa) changing to DMA kernel invocation.
        q.single_task(DMA_convert_data<float, Conv1DInputPipe, kinputSz>{vals_device_ptr});
        q.single_task(Myproject{});
        // hls-fpga-machine-learning convert output
        // nnet::convert_data_back<Layer4OutPipe, float, N_OUT_4>(q, outputs);
        // (haoyanwa) changing to DMA kernel invocation.
        q.single_task(DMA_convert_data_back<Layer4OutPipe, float, kOutputSz>{output_device_ptr}).wait();
        q.memcpy(outputs, output_device_ptr, kOutputSz * sizeof(float)).wait();
    
        // After receiving all the output, write a `true` into `StopPipe` to instruct the kernel to break
        // out of its main loop.
        // StopPipe::write(q, true);
        for (auto outval : outputs) {
            std::cout << outval << " ";
        }
        std::cout << std::endl;

        for (auto outval : outputs) {
            fout << outval << " ";
        }
        fout << std::endl;
    }
    q.wait();

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
