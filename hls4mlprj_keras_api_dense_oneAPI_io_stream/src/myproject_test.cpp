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

    std::cout << "Running on device: " << device.get_info<sycl::info::device::name>().c_str() << std::endl;

    constexpr size_t kNumBatch = 300;
    constexpr size_t kinputSz = kNumBatch * N_INPUT_1_1;
    constexpr size_t kOutputSz = kNumBatch * N_LAYER_2;
    std::cout << "INFO: Using default input with " << kNumBatch << " batch" << std::endl;

    try
    {
        InputBeatT ctype;

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
            DenseInputPipe::write(q, ctype);
        }

        constexpr size_t srcTypeSize = std::tuple_size<layer2_t>{};
        double buffer[kOutputSz];
        layer2_t octype;

        for (size_t i = 0; i < kOutputSz / srcTypeSize; i++)
        {
            octype = Layer2OutPipe::read(q);
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

    return 0;
}
