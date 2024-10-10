#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "defines.h"

#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_stream.h"

// hls-fpga-machine-learning insert layer-config
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 16;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b2_t bias_t;
    typedef w2_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config2 : nnet::conv2d_config {
    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned n_chan = 1;

    static const unsigned out_height = 28;
    static const unsigned out_width = 28;

    static const unsigned n_filt = 16;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned impl_filt_height = 3;
    static const unsigned impl_filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;

    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv2d_implementation implementation = nnet::conv2d_implementation::im2col;

    typedef model_default_t accum_t;
    typedef b2_t bias_t;
    typedef w2_t weight_t;
    typedef config2_mult mult_config;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 12544;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef conv2d_12_relu_table_t table_t;
};

struct config4 : nnet::pooling2d_config {
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;

    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;

    static const unsigned in_height = 28;
    static const unsigned in_width = 28;
    static const unsigned out_height = 14;
    static const unsigned out_width = 14;

    static const unsigned n_filt = 16;
    static const unsigned n_chan = 16;

    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b5_t bias_t;
    typedef w5_t weight_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5 : nnet::conv2d_config {
    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned n_chan = 16;

    static const unsigned out_height = 14;
    static const unsigned out_width = 14;

    static const unsigned n_filt = 32;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned impl_filt_height = 3;
    static const unsigned impl_filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;

    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;

    static const unsigned reuse_factor = 1;
    static const unsigned parallelization_factor = 1;
    static const bool store_weights_in_bram = false;

    static const nnet::conv2d_implementation implementation = nnet::conv2d_implementation::im2col;

    typedef model_default_t accum_t;
    typedef b5_t bias_t;
    typedef w5_t weight_t;
    typedef config5_mult mult_config;
};

struct relu_config6 : nnet::activ_config {
    static const unsigned n_in = 6272;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef conv2d_13_relu_table_t table_t;
};

struct config7 : nnet::pooling2d_config {
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;

    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;
    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;

    static const unsigned in_height = 14;
    static const unsigned in_width = 14;
    static const unsigned out_height = 7;
    static const unsigned out_width = 7;

    static const unsigned n_filt = 32;
    static const unsigned n_chan = 32;

    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = false;

    static const nnet::Pool_Op pool_op = nnet::Max;
    typedef model_default_t accum_t;
};

struct config9 : nnet::dense_config {
    static const unsigned n_in = 1568;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 15680;
    static const bool store_weights_in_bram = false;

    static const unsigned rf_pad = 0;
    static const unsigned bf_pad = 0;

    static const unsigned reuse_factor = 1;
    static const unsigned compressed_block_factor = DIV_ROUNDUP(n_nonzeros, reuse_factor);
    static const unsigned reuse_factor_rounded = reuse_factor + rf_pad;
    static const unsigned block_factor = DIV_ROUNDUP(n_in*n_out, reuse_factor);
    static const unsigned block_factor_rounded = block_factor + bf_pad;
    static const unsigned multiplier_factor = MIN(n_in, reuse_factor);
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in*n_out, multiplier_factor);
    static const unsigned multiplier_scale = multiplier_limit/n_out;

    typedef model_default_t accum_t;
    typedef b9_t bias_t;
    typedef w9_t weight_t;
    typedef layer9_index index_t;

    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct softmax_config10 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef dense_6_softmax_exp_table_t exp_table_t;
    typedef dense_6_softmax_inv_table_t inv_table_t;
};


#endif
