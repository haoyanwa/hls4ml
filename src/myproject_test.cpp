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

    if (fin.is_open() && fpr.is_open()) {
        std::vector<std::vector<float>> predictions;
        unsigned int iteration = 0;
        for (; std::getline(fin, iline) && std::getline(fpr, pline); iteration++) {
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Processing input " << iteration << std::endl;
            }

            std::vector<float> in;
            std::vector<float> pr;
            float current;

            std::stringstream ssin(iline);
            while (ssin >> current) {
                in.push_back(current);
            }

            std::stringstream sspred(pline);
            while (sspred >> current) {
                pr.push_back(current);
            }

            // hls-fpga-machine-learning insert data
            float vals[N_INPUT_1_1*N_INPUT_2_1]; 
            for (int j = 0 ; j < N_INPUT_1_1*N_INPUT_2_1 ; j++) {
                vals[j] = in[j]; 
            }
            nnet::convert_data<float, Conv1DInputPipe, N_INPUT_1_1*N_INPUT_2_1>(q, vals);

            q.single_task(Myproject{});

            // hls-fpga-machine-learning convert output
            float outputs[N_OUT_4];
            nnet::convert_data_back<Layer4OutPipe, float, N_OUT_4>(q, outputs);

            std::copy(pr.cbegin(), pr.cend(), predictions.back().begin());

            for (auto outval : outputs) {
                fout << outval << " ";
            }
            fout << std::endl;
            if (iteration % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for (auto predval : pr) {
                    std::cout << predval << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                for (auto outval : outputs) {
                    std::cout << outval << " ";
                }
                std::cout << std::endl;
            }
        }
        fin.close();
        fpr.close();
    } else {
        const unsigned int num_iterations = 2;
        std::cout << "INFO: Unable to open input/predictions file, using default input with " << num_iterations
                  << " invocations." << std::endl;

        constexpr size_t kinputSz = N_INPUT_1_1*N_INPUT_2_1; 
        // hls-fpga-machine-learning insert zero
        // (haoyanwa) change the input and output to device ptr.
        float *vals_device_ptr = sycl::malloc_device<float>(kinputSz, q);
        float *output_device_ptr = sycl::malloc_device<float>(N_OUT_4, q);

        // Temp array to buffer data on host side.
        float vals[kinputSz]; 
        float outputs[N_OUT_4];

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
            q.single_task(DMA_convert_data<float, Conv1DInputPipe, N_INPUT_1_1*N_INPUT_2_1>{vals_device_ptr});
            q.single_task(Myproject{});
            // hls-fpga-machine-learning convert output
            // nnet::convert_data_back<Layer4OutPipe, float, N_OUT_4>(q, outputs);
            // (haoyanwa) changing to DMA kernel invocation.
            q.single_task(DMA_convert_data_back<Layer4OutPipe, float, N_OUT_4>{output_device_ptr});
            q.memcpy(outputs, output_device_ptr, N_OUT_4 * sizeof(float)).wait();
        
            // After receiving all the output, write a `true` into `StopPipe` to instruct the kernel to break
            // out of its main loop.
            StopPipe::write(q, true);
            std::cout << "Output DMA Wrote stop to pipe";
            for (auto outval : outputs) {
                std::cout << outval << " ";
            }
            std::cout << std::endl;

            for (auto outval : outputs) {
                fout << outval << " ";
            }
            fout << std::endl;
        }
    }
    q.wait();

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
