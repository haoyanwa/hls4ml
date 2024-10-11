#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_types.h"


namespace nnet {

// Note:  DataPack logic removed, at least in the initial version

template <class data_pipe, class res_pipe, typename CONFIG_T>
[[intel::use_stall_enable_clusters]] void dense_resource_stream(const typename CONFIG_T::weight_t weights, const typename CONFIG_T::bias_t biases) {
    using namespace nnet;
    using DataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;
    using ResT = typename ExtractDataType<typename ExtractPipeType<res_pipe>::value_type>::value_type;
    
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type resbeat;
    // [[intel::fpga_register]] ResT res;

    bool keep_going = true;
    bool did_read_input;
    [[intel::initiation_interval(1)]]
    while (keep_going) {
        did_read_input = false;
        [[intel::fpga_register]] auto databeat = data_pipe::read(did_read_input);

        if (did_read_input) {
            dense_resource<DataT, ResT, CONFIG_T>(databeat.data, resbeat.data, weights, biases);

            resbeat.sop = databeat.sop;
            resbeat.eop = databeat.eop;

            res_pipe::write(resbeat);
            keep_going = !databeat.eop;
        }
    }
}

} // namespace nnet

#endif
