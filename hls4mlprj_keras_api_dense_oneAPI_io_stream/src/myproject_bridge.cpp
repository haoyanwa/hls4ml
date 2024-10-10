#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

#include "exception_handler.hpp"

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

class MyprojectClassFloat_db9fe47a;

// Wrapper of top level function for Python bridge
void myproject_float(
    float Dense_input[N_INPUT_1_1],
    float layer4_out[N_LAYER_2]
) {
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    static sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

    nnet::convert_data<float, DenseInputPipe, N_INPUT_1_1>(q, Dense_input);
    q.single_task<MyprojectClassFloat_db9fe47a>(Myproject{});
    nnet::convert_data_back<Layer4OutPipe, float, N_LAYER_2>(q, layer4_out);

    q.wait();
}

class MyprojectClassDouble_db9fe47a;

void myproject_double(
    double Dense_input[N_INPUT_1_1],
    double layer4_out[N_LAYER_2]
) {
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    static sycl::queue q(selector, fpga_tools::exception_handler, sycl::property::queue::enable_profiling{});

    nnet::convert_data<double, DenseInputPipe, N_INPUT_1_1>(q, Dense_input);
    q.single_task<MyprojectClassDouble_db9fe47a>(Myproject{});
    nnet::convert_data_back<Layer4OutPipe, double, N_LAYER_2>(q, layer4_out);

    q.wait();
}
}

#endif
