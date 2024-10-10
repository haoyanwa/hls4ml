#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 28
#define OUT_WIDTH_2 28
#define N_FILT_2 16
#define OUT_HEIGHT_2 28
#define OUT_WIDTH_2 28
#define N_FILT_2 16
#define OUT_HEIGHT_4 14
#define OUT_WIDTH_4 14
#define N_FILT_4 16
#define OUT_HEIGHT_5 14
#define OUT_WIDTH_5 14
#define N_FILT_5 32
#define OUT_HEIGHT_5 14
#define OUT_WIDTH_5 14
#define N_FILT_5 32
#define OUT_HEIGHT_7 7
#define OUT_WIDTH_7 7
#define N_FILT_7 32
#define N_SIZE_0_8 1568
#define N_LAYER_9 10
#define N_LAYER_9 10

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<16,6,true>, 784*1> input_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef nnet::array<ac_fixed<37,17,true>, 12544*1> conv2d_12_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 144*1> w2_t;
typedef nnet::array<ac_fixed<16,6,true>, 16*1> b2_t;
typedef nnet::array<ac_fixed<16,6,true>, 12544*1> layer3_t;
typedef ac_fixed<18,8,true> conv2d_12_relu_table_t;
typedef nnet::array<ac_fixed<16,6,true>, 3136*1> layer4_t;
typedef nnet::array<ac_fixed<41,21,true>, 6272*1> conv2d_13_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 4608*1> w5_t;
typedef nnet::array<ac_fixed<16,6,true>, 32*1> b5_t;
typedef nnet::array<ac_fixed<16,6,true>, 6272*1> layer6_t;
typedef ac_fixed<18,8,true> conv2d_13_relu_table_t;
typedef nnet::array<ac_fixed<16,6,true>, 1568*1> layer7_t;
typedef nnet::array<ac_fixed<44,24,true>, 10*1> dense_6_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 15680*1> w9_t;
typedef nnet::array<ac_fixed<16,6,true>, 10*1> b9_t;
typedef ac_int<1, false> layer9_index;
typedef nnet::array<ac_fixed<16,6,true>, 10*1> result_t;
typedef ac_fixed<18,8,true> dense_6_softmax_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> dense_6_softmax_exp_table_t;
typedef ac_fixed<18,8,true,AC_RND,AC_SAT> dense_6_softmax_inv_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
