#include <vector>
#include <string>
#include <xcl2/xcl2.hpp>
#include <xcl2/xcl2.hpp>
#include "ap_fixed.h"

typedef ap_ufixed<19, 11> data_in;
typedef ap_ufixed<20, 12> data_out;

int cumprod(std::vector<int>);

void print_vec(std::vector<int>);

void print_vec(std::vector<bool>);

void pad_begin(std::vector<int> &, int, int);

void pad_end(std::vector<int> &, int, int);

std::vector<int> sub_vecs(std::vector<int>, std::vector<int>);

std::vector<int> add_vecs(std::vector<int>, std::vector<int>);

std::vector<int> squeeze(std::vector<int>);

std::vector<int> broadcast(std::vector<int>, std::vector<int>);

void negotiate_shapes(std::vector<int>, std::vector<int>, std::vector<int>);

std::vector<int> stride_from_shape(std::vector<int>);

std::vector<int> filter_on_squeeze(std::vector<int>, std::vector<int>);

std::vector<int> zero_on_squeeze(std::vector<int>, std::vector<int>);

int collect_linear_offset(std::vector<int>, std::vector<int>, std::vector<int>);

std::vector<int> rebuild_stride(std::vector<int>, std::vector<int>, int);

std::vector<int> rebuild_offset(std::vector<int>, std::vector<int>, std::vector<int>, int);

void negotiate_strides(	
					std::vector<int>, 
					std::vector<int> &, 
					std::vector<int> &, 
					std::vector<int> &,
					std::vector<int>, 
					std::vector<int> &,
					std::vector<int> &, 
					std::vector<int> &,
					int &, 
					int &, 
					int &,
					int &, 
					int &
					);

int next_largets_factor_2(int);

void run_1d_kernel(
					std::string,
					std::vector<double *> &,
					std::vector<double *> &,
					std::vector<int>,
					std::vector<int>,
					std::vector<int>,
					std::vector<int>,
					std::vector<int>,
					std::vector<int>,
					std::vector<cl::Device> &,
					cl::Context &,
					cl::Program::Binaries &,
					cl::CommandQueue &);

void run_broadcast_kernel(std::string,
							std::vector<data_in *> &,
							std::vector<data_out *> &,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<cl::Device> &,
							cl::Context &,
							cl::Program::Binaries &,
							cl::CommandQueue &);

void run_gtsv(int, std::vector<double *> &, std::vector<cl::Device> &, cl::Context &, cl::Program::Binaries &, cl::CommandQueue &);

void run_where_kernel(std::string, 
							std::vector<double *> &,
							std::vector<double *> &,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<int>,
							std::vector<cl::Device> &,
							cl::Context &,
							cl::Program::Binaries &,
							cl::CommandQueue &);

