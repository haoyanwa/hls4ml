#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_types.h"

#include <sycl/ext/intel/prototype/pipes_ext.hpp>


// Helper to extract DataT from StreamingBeat
template <typename T> struct ExtractDataType { typedef T value_type; };

template <typename DataT, bool EnableSOP, bool EnableEmpty>
struct ExtractDataType<sycl::ext::intel::experimental::StreamingBeat<DataT, EnableSOP, EnableEmpty>> {
    typedef DataT value_type;
};

namespace nnet {

// Note:  DataPack logic removed, at least in the initial version
template <class data_pipe, class res_pipe, typename CONFIG_T>
void dense_resource_stream(const typename CONFIG_T::weight_t weights, const typename CONFIG_T::bias_t biases) {

    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res;

    using PipeDataT = typename ExtractDataType<typename ExtractPipeType<data_pipe>::value_type>::value_type;

    bool keep_going = true;
    while (keep_going) {
        [[intel::fpga_register]] auto databeat = data_pipe::read();
        dense_resource<PipeDataT, typename ExtractPipeType<res_pipe>::value_type,
                   CONFIG_T>(databeat.data, res, weights, biases);
        res_pipe::write(res);
        keep_going = !databeat.eop;
    }

}

} // namespace nnet

#endif
