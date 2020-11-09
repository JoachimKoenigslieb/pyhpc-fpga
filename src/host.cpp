#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include "cnpy.h"
#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xnpy.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"

#include "xcl2.cpp"

#define X 32
#define Y 32
#define Z 4

using namespace xt::placeholders; //enables xt::range(1, _) syntax. eqiv. to [1:] syntax in numpy 

int cumprod(std::vector<int> v){
	int i = 1;
	for (int j=0; j<v.size(); j++){
		i = i * v[j];
	}
	return i;
}

void print_vec(std::vector<int> v){
	for (auto &elm : v) {
		std::cout << elm << ", ";
	};
	std::cout << std::endl;
}

void print_vec(std::vector<bool> v){
	int len = v.size();
	for (int i=0; i<len; i++) {
		std::cout << v[i] << ", ";
	};
	std::cout << std::endl;
}

void pad_begin(std::vector<int> &vec, int val, int to_len){
	int vec_len = vec.size();
	for (int i=0; i<(to_len - vec_len); i++){
		vec.insert(vec.begin(), val);
	}
}

void push_right(std::vector<int> &vec, int num){
	int vec_size = vec.size();
	vec.erase(vec.end() - num, vec.end());
	pad_begin(vec, 0, vec_size);
}

int count_trailing_ones(std::vector<int> &vec){
	int ones=0;
	int vec_size = vec.size();
	for (int i=vec_size-1; i>-1; i--){
		if (vec[i] == 1){
			ones++;
		} else {
			break;
		}
	}
	return ones;
}


void calculate_shapes_for_broadcasting(std::vector<int> &shape_A, std::vector<int> &shape_B, std::vector<int> requested_shape_output)
{
	int len_O = requested_shape_output.size();
	int len_A = shape_A.size();
	int len_B = shape_B.size();

	std::cout << "A has len " << len_A << " B has len: " << len_B << " requested output has len " << len_O << std::endl;

	pad_begin(shape_A, 1, len_O);
	pad_begin(shape_B, 1, len_O);

	// loop trough and check if we can broadcast
	int dim_A, dim_B, dim_O, naive_output;

	for (int i = 0; i < len_O; i++)
	{
		dim_A = shape_A[i];
		dim_B = shape_B[i];
		dim_O = requested_shape_output[i];

		if (!(dim_A == dim_B || dim_A == 1 || dim_B == 1))
		{
			std::cout << "!!!\n!!!\nERROR: You tried to broadcast shapes that does not broadcast\n!!!\n!!!\n!!!\n"; //do proper error handling
		}

		naive_output = std::max(dim_A, dim_B); //this is max of inputs shapes. should be either 1, or equal to requested shape!
		std::cout << "broadcasting the inputs gives: " << naive_output << ". We are requesting: " << dim_O << std::endl;

		if (!(dim_O == naive_output || naive_output == 1))
		{
			std::cout << "!!!\n!!!\nERROR: You tried to broadcast into a shape that can not be broadcast from the inputs!\n!!!\n!!!\n!!!\n";
		}
	}
}

int vec_sum(std::vector<bool> v){
	int s=0;
	int dims = v.size();
	for (int i=0; i<dims; i++){
		s+=v[i];
	}
	return s;
}

std::vector<int> get_stride_from_shape(std::vector<int> shape){
	int dims = shape.size();
	std::vector<int> stride(dims);
	// stride for dimension i is cumulative product of size of dimensions higher than i.

	for (int i=0; i<dims; i++){
		stride[i] = cumprod(std::vector<int>(shape.begin() + i + 1, shape.end()));
		if (shape[i] == 1){
			stride[i] = 0;
		}
	}

	return stride;
}

// Memory alignment
template <typename T>
T *aligned_alloc(std::size_t num)
{
	void *ptr = nullptr;
	if (posix_memalign(&ptr, 4096, num * sizeof(T)))
	{
		throw std::bad_alloc();
	}
	return reinterpret_cast<T *>(ptr);
}

// Compute time difference
unsigned long diff(const struct timeval *newTime, const struct timeval *oldTime)
{
	return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

void run_kernel(std::string kernel_name, int kernel_size, std::vector<double *> &inputs, std::vector<double *> &outputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
        // this is a helper function to execute a kernel.
        {
                cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
                cl::Kernel kernel(program, kernel_name.data());

                std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

                int num_in = inputs.size();
                int num_out = outputs.size();

                std::vector<cl::Buffer> in_buffers(num_in);
                std::vector<cl::Buffer> out_buffers(num_out);

                for (int i = 0; i < num_in; i++)
                {
                        in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * kernel_size, inputs[i]);
                }

                for (int i = 0; i < num_out; i++)
                {
                        out_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double) * kernel_size, outputs[i]);
                }

                std::cout << "INFO: Buffers has been created" << std::endl;

                kernel.setArg(0, kernel_size); //we need to set arg0 to size!

                for (int i = 0; i < num_in; i++)
                {
                        kernel.setArg(i + 1, in_buffers[i]);
                        q.enqueueMigrateMemObjects({in_buffers[i]}, 0);
                }

                for (int i = 0; i < num_out; i++)
                {
                        kernel.setArg(i + num_in + 1, out_buffers[i]);
                }

                std::cout << "INFO: Arguments has been set" << std::endl;

                std::cout << "INFO: Migrated to device" << std::endl;

                q.enqueueTask(kernel);

                q.finish();

                q.enqueueMigrateMemObjects({out_buffers[0]}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

                std::cout << "INFO: Migrated to back to host" << std::endl;

                q.finish();
        }
}


std::vector<int> sub_vecs(std::vector<int> A, std::vector<int> B){
	int dims = A.size();
	std::vector<int> res(dims);
	for (int i=0; i<dims; i++){
		res[i] = A[i]-B[i];
	}
	return res;
}

std::vector<int> add_vecs(std::vector<int> A, std::vector<int> B){
	int dims = A.size();
	std::vector<int> res(dims);
	for (int i=0; i<dims; i++){
		res[i] = A[i]+B[i];
	}
	return res;
}

std::vector<bool> singleton_dimension_mask(std::vector<int> v){
	int dimensions = v.size();
	std::vector<bool> mask(dimensions);
	for (int i=0; i<dimensions; i++){
		if (v[i] == 1){
			mask[i] = true;
		} else {
			mask[i] = false;
		}
	}
	return mask;
}

void remove_leading_zeros(std::vector<int> &v){
	int dims = v.size();
	int leading_zeros = 0;
	for (int i=0; i<dims; i++){
		if (v[i] == 0){
			leading_zeros++;
		} else {
			break;
		}
	} 
	v = std::vector<int>(v.begin() + leading_zeros, v.end());
}

bool mask_equality(std::vector<bool> mask_A, std::vector<bool> mask_B){
	int dims = mask_A.size();

	for (int i=0; i<dims; i++){
		if (mask_A[i] != mask_B[i]){
			return false;
		}
	}
	return true;
}

void rebuild_stride(std::vector<int> &stride, std::vector<bool> input_singleton_mask, std::vector<bool> output_singleton_mask){
	std::vector<int> stride_pool;
	int dimensions = output_singleton_mask.size();

	
	std::cout << "Attempting to rebuild strides for stride vector ";
	print_vec(stride);
	std::cout << "Input mask is ";
	print_vec(input_singleton_mask);
	std::cout << "Output mask is ";
	print_vec(output_singleton_mask);
	std::cout << "Building stride pool...\n";
	

	for (int i=0; i<dimensions; i++){
		if (!input_singleton_mask[i]){
			stride_pool.push_back(stride[i]);
		}
	}

	
	std::cout << "stride pool is: ";
	print_vec(stride_pool);
	std::cout << "rebulding stride...\n";
	
	std::vector<int> output_mask_where;
	for (int i=0; i<dimensions; i++){
		if (!output_singleton_mask[i]){
			output_mask_where.push_back(i); //All the places we could possible put a stride. We put it into the most-leading dimensions.
		}
	}

	std::cout << "available indexes to put new strides: ";
	print_vec(output_mask_where);

	int available_strides = stride_pool.size();
	for (int i=0; i<available_strides; i++){
		stride[output_mask_where.back()] = stride_pool.back();
		stride_pool.pop_back();
		output_mask_where.pop_back();
	}

	std::cout << "new stride:\n";
	print_vec(stride);
}

void run_broadcast_kernel(std::string kernel_name,
							std::vector<double *> &inputs,
							std::vector<double *> &outputs,
							std::vector<int> A_shape,
							std::vector<int> B_shape,
							std::vector<int> output_shape,
							std::vector<int> A_offset,
							std::vector<int> B_offset,
							std::vector<int> output_offset,
							std::vector<int> A_offset_end,
							std::vector<int> B_offset_end,
							std::vector<int> output_offset_end,
							std::vector<cl::Device> &devices,
							cl::Context &context,
							cl::Program::Binaries &bins,
							cl::CommandQueue &q)
{
	// this is a helper function to execute a kernel.
	{
		//setup program and kernel:
		cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
		cl::Kernel kernel(program, kernel_name.data());
		std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

		//num inputs outputs
		int num_in = inputs.size();
		int num_out = outputs.size();

		//setup buffers:
		std::vector<cl::Buffer> in_buffers(num_in);
		std::vector<cl::Buffer> in_stride_buffers(num_in);
		std::vector<cl::Buffer> in_offset_buffers(num_in);

		cl::Buffer out_buffer;
		cl::Buffer out_shape_buffer;
		cl::Buffer out_stride_buffer;
		cl::Buffer out_offset_buffer;
		cl::Buffer out_offset_end_buffer;

		std::cout << "INFO: Buffers initialized. Calculating sizes, shapes, strides etc...\n";

		//calculate size variables
		int len_A, len_B, len_out;
		len_A = A_shape.size(),
		len_B = B_shape.size(),
		len_out = output_shape.size();
		int max_in = std::max(len_A, len_B);
		int dimensions = std::max(max_in, len_out);
		pad_begin(A_shape, 1, dimensions); //add singleton dimensions
		pad_begin(B_shape, 1, dimensions);
		pad_begin(output_shape, 1, dimensions);
		pad_begin(A_offset, 0, dimensions); //zero pad offsets if not defined
		pad_begin(B_offset, 0, dimensions);
		pad_begin(output_offset, 0, dimensions);
		pad_begin(A_offset_end, 0, dimensions);
		pad_begin(B_offset_end, 0, dimensions);
		pad_begin(output_offset_end, 0, dimensions);

		std::cout << "INFO: Basic size stuff calculated. Rebuiling for edge cases...\n";

		//calculate view shapes:
		std::vector<int> A_view_shape, B_view_shape, output_view_shape; 
		A_view_shape = sub_vecs(add_vecs(A_shape, A_offset_end), A_offset); //A_offset_end is a negative index indicating how many to take from the end. 
		B_view_shape = sub_vecs(add_vecs(B_shape, B_offset_end), B_offset); //A_offset_end is a negative index indicating how many to take from the end. 
		output_view_shape = sub_vecs(add_vecs(output_shape, output_offset_end), output_offset); //A_offset_end is a negative index indicating how many to take from the end. 

		for (int i=0; i<dimensions; i++){
			A_offset[i] -= output_offset[i];
			B_offset[i] -= output_offset[i];
		}

		std::cout << "INFO: view shapes found...\n";

		std::vector<int> data_sizes_input = {cumprod(A_shape), cumprod(B_shape)};
		int data_size_output = cumprod(output_shape);

		std::vector<int> A_stride, B_stride, output_stride;

		//calculate strides and offsets
		output_stride = get_stride_from_shape(output_shape);
		A_stride = get_stride_from_shape(A_shape);
		B_stride = get_stride_from_shape(B_shape);

		std::cout << "INFO: naive strides found...\n";

		//Find possible singleton dimensions in inputs and output
		std::vector<bool> output_view_singleton_dimensions = singleton_dimension_mask(output_view_shape);
		std::vector<bool> A_view_singleton_dimensions = singleton_dimension_mask(A_view_shape);
		std::vector<bool> B_view_singleton_dimensions = singleton_dimension_mask(B_view_shape);
		
		std::vector<bool> A_singleton_dimensions = singleton_dimension_mask(A_shape);
		std::vector<bool> B_singleton_dimensions = singleton_dimension_mask(B_shape);
		std::cout << "INFO: Singleton masks found...\n";

		//Output is correctly strided by assumption: If you supply correct dimension of output i will loop trough every entry exactly once with strides calculated from you supplied shape.
		//If inputs is not the same dimensions as output, we get in trouble with naive strides. We instead must rearrange the strides of inputs such that:
		//1: Strides corresponding to singelton dimensions gets removed from "stride pool"
		//2: stride pool is then arranged in-order skipping any singleton dimensions in the output. 

		if (!mask_equality(A_singleton_dimensions, A_view_singleton_dimensions) || !mask_equality(B_singleton_dimensions, B_view_singleton_dimensions)){
			rebuild_stride(A_stride, A_view_singleton_dimensions, output_view_singleton_dimensions);
			rebuild_stride(B_stride, B_view_singleton_dimensions, output_view_singleton_dimensions);
		}

		std::cout << "INFO: Rebuild strides taking singletons dimensions into account";

		//create "looping over" vectors (seems stupid to it this way...)
		std::vector<std::vector<int>> input_strides = {A_stride, B_stride};
		std::vector<std::vector<int>> input_offset = {A_offset, B_offset};

		//print debug info:
		std::cout << "A_shapes: ";
		print_vec(A_shape);
		std::cout << "B_shapes: ";
		print_vec(B_shape);
		std::cout << "Out_shapes ";
		print_vec(output_shape);

		std::cout << "A_view_shapes: ";
		print_vec(A_view_shape);
		std::cout << "B_view_shapes: ";
		print_vec(B_view_shape);
		std::cout << "output_view_shapes ";
		print_vec(output_view_shape);

		std::cout << "Singleton dimenions A:";
		print_vec(A_view_singleton_dimensions);
		std::cout << "Singleton dimenions B:";
		print_vec(B_view_singleton_dimensions);
		std::cout << "Singleton dimenions output:";
		print_vec(output_view_singleton_dimensions);

		std::cout << "A_offset: ";
		print_vec(A_offset);
		std::cout << "B_offset: ";
		print_vec(B_offset);
		std::cout << "out_offset: ";
		print_vec(output_offset);


		std::cout << "A_offset_end: ";
		print_vec(A_offset_end);		
		std::cout << "B_offset_end: ";
		print_vec(B_offset_end);		
		std::cout << "out_offset_end: ";
		print_vec(output_offset_end);

		std::cout << "A strides: ";
		print_vec(input_strides[0]);
		std::cout << "B strides: ";
		print_vec(input_strides[1]);
		std::cout << "O strides: ";
		print_vec(output_stride);

		std::cout << "Input size: " << data_sizes_input[0] << ", " << data_sizes_input[1] << std::endl;
		std::cout << "Output size: " << data_size_output << std::endl;

		// write to buffers
		for (int i = 0; i < num_in; i++){
			in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * data_sizes_input[i], inputs[i]);
			in_stride_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_strides[i].data());
			in_offset_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_offset[i].data());
		}

		out_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * data_size_output, outputs[0]);
		out_shape_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_shape.data());
		out_stride_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_stride.data());
		out_offset_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset.data());
		out_offset_end_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset_end.data());

		std::cout << "INFO: Buffers has been created" << std::endl;

		//set inputs.
		for (int i = 0; i < num_in; i++){
			kernel.setArg(i, in_buffers[i]);
			q.enqueueMigrateMemObjects({in_buffers[i]}, 0);
		}

		std::cout << "INFO: inputs has been set." <<std::endl;

		//set strides
		for (int i = num_in; i < (2 * num_in); i++){
			kernel.setArg(i, in_stride_buffers[i-num_in]);
			q.enqueueMigrateMemObjects({in_stride_buffers[i-num_in]}, 0);
		}

		std::cout << "INFO: strides has been set." <<std::endl;

		//set offsets
		for (int i = (2 * num_in); i < (3 * num_in); i++){
			kernel.setArg(i, in_offset_buffers[i-2*num_in]);
			q.enqueueMigrateMemObjects({in_offset_buffers[i-2*num_in]}, 0);
		}

		std::cout << "INFO: offests has been set." <<std::endl;

		//set outputs
		kernel.setArg(3 * num_in, out_buffer);			  //arg6
		kernel.setArg(3 * num_in + 1, out_shape_buffer);	  //arg7
		kernel.setArg(3 * num_in + 2, out_stride_buffer); //arg8
		kernel.setArg(3 * num_in + 3, out_offset_buffer);		  //arg9
		kernel.setArg(3 * num_in + 4, out_offset_end_buffer);  //arg10

		q.enqueueMigrateMemObjects({out_shape_buffer}, 0);
		q.enqueueMigrateMemObjects({out_stride_buffer}, 0);
		q.enqueueMigrateMemObjects({out_offset_buffer}, 0);
		q.enqueueMigrateMemObjects({out_offset_end_buffer}, 0);

		std::cout << "INFO: outputs has been set\n";

		q.finish();

		std::cout << "INFO: Arguments are set\n";

		q.enqueueTask(kernel);
		q.finish();
		q.enqueueMigrateMemObjects({out_buffer}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

		std::cout << "INFO: Migrated to back to host" << std::endl;

		q.finish();
	}
}


// Arguments parser
class ArgParser
{
public:
	ArgParser(int &argc, const char **argv)
	{
		for (int i = 1; i < argc; ++i)
			mTokens.push_back(std::string(argv[i]));
	}
	bool getCmdOption(const std::string option, std::string &value) const
	{
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
		if (itr != this->mTokens.end() && ++itr != this->mTokens.end())
		{
			value = *itr;
			return true;
		}
		return false;
	}

private:
	std::vector<std::string> mTokens;
};

int main(int argc, const char *argv[])
{
	// Init of FPGA device
	std::string xclbin_path = "./kernels.xclbin";
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
	devices.resize(1);

	int size_4d = X * Y * Z * 3;
	int size_3d = X * Y * Z;
	int size_2d = X * Y;
	int size_1d_vert = Z;
	int size_1d_hoz = X; //assert(X == Y)!!!

	xt::xarray<double> u = xt::load_npy<double>("../src/numpy_files/u.npy");
	xt::xarray<double> v = xt::load_npy<double>("../src/numpy_files/v.npy");
	xt::xarray<double> w = xt::load_npy<double>("../src/numpy_files/w.npy");
	xt::xarray<double> maskU = xt::load_npy<double>("../src/numpy_files/maskU.npy");
	xt::xarray<double> maskV = xt::load_npy<double>("../src/numpy_files/maskV.npy");
	xt::xarray<double> maskW = xt::load_npy<double>("../src/numpy_files/maskW.npy");
	xt::xarray<double> dxt = xt::load_npy<double>("../src/numpy_files/dxt.npy");
	xt::xarray<double> dxu = xt::load_npy<double>("../src/numpy_files/dxu.npy");
	xt::xarray<double> dyt = xt::load_npy<double>("../src/numpy_files/dyt.npy");
	xt::xarray<double> dyu = xt::load_npy<double>("../src/numpy_files/dyu.npy");
	xt::xarray<double> dzt = xt::load_npy<double>("../src/numpy_files/dzt.npy");
	xt::xarray<double> dzw = xt::load_npy<double>("../src/numpy_files/dzw.npy");
	xt::xarray<double> cost = xt::load_npy<double>("../src/numpy_files/cost.npy");
	xt::xarray<double> cosu = xt::load_npy<double>("../src/numpy_files/cosu.npy");
	xt::xarray<double> kbot = xt::load_npy<double>("../src/numpy_files/kbot.npy");
	xt::xarray<double> forc_tke_surface = xt::load_npy<double>("../src/numpy_files/forc_tke_surface.npy");
	xt::xarray<double> kappaM = xt::load_npy<double>("../src/numpy_files/kappaM.npy");
	xt::xarray<double> mxl = xt::load_npy<double>("../src/numpy_files/mxl.npy");
	xt::xarray<double> forc = xt::load_npy<double>("../src/numpy_files/forc.npy");
	xt::xarray<double> tke = xt::load_npy<double>("../src/numpy_files/tke.npy");
	xt::xarray<double> dtke = xt::load_npy<double>("../src/numpy_files/dtke.npy");
	xt::xarray<double> flux_east = xt::zeros<double>({X, Y, Z});
	xt::xarray<double> flux_north = xt::zeros<double>({X, Y, Z});
	xt::xarray<double> flux_top = xt::zeros<double>({X, Y, Z});
	xt::xarray<double> sqrttke = xt::empty<double>({X, Y, Z});
	xt::xarray<double> a_tri = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> b_tri = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> b_tri_edge = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> c_tri = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> d_tri = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> delta = xt::zeros<double>({X-4, Y-4, Z});
	xt::xarray<double> ks = xt::zeros<double>({X-4, Y-4});

	int tau = 0;
	double taup1 = 1.;
	double taum1 = 2.;
	double dt_tracer = 1.;
	double dt_mom = 1.;
	double dt_tke = 1.;
	double AB_eps = 0.1;
	double alpha_tke = 1.;
	double c_eps = 0.7;
	double K_h_tke = 2000.;

	std::vector<double *> inputs;
	std::vector<double *> outputs;

	int tmp_int;

	xt::xarray<double> one = xt::ones<double>({1});
	xt::xarray<double> zero = xt::zeros<double>({1});
	xt::xarray<double> half = xt::ones<double>({1}) * 0.5;
	xt::xarray<double> two = xt::ones<double>({1}) * 2;
	
	//sqrttke
	inputs = {tke.data(), zero.data()};
	outputs = {sqrttke.data()};
	run_broadcast_kernel("max4d", inputs, outputs, 
		{X, Y, Z, 3}, {1}, {X, Y, Z},			//shapes
		{0, 0, 0, 0}, {0}, {0, 0, 0,},			//start index
		{0, 0, 0, -2}, {0}, {0, 0, 0}, 			//negativ end index
		devices, context, bins, q);

	inputs = {sqrttke.data()};
	outputs = {sqrttke.data()};
	run_kernel("vsqrt", size_3d, inputs, outputs, devices, context, bins, q);

	//kbot
	inputs = {kbot.data(), one.data()};
	outputs = {ks.data()};
	run_broadcast_kernel("sub2d", inputs, outputs, 
		{X, Y}, {1}, {X-4, Y-4,},			//shapes
		{2, 2}, {0}, {0, 0},			//start index
		{-2, -2}, {0}, {0, 0}, 			//negativ end index
		devices, context, bins, q);

	//delta
	inputs = {kappaM.data(), kappaM.data()};
	outputs = {delta.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X, Y, Z}, {X,Y,Z}, {X-4, Y-4, Z},			//shapes
		{2, 2, 0}, {2, 2, 1}, {0, 0, 0,},			//start index
		{-2, -2, -1}, {-2, -2, 0}, {0, 0, -1}, 		//negativ end index
		devices, context, bins, q);

	inputs = {half.data(), delta.data()};
	outputs = {delta.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0,}, {0, 0, 0}, {0, 0, 0,},				//start index
		{0,}, {0, 0, -1}, {0, 0, -1}, 				//negativ end index
		devices, context, bins, q);

	inputs = {delta.data(), dzt.data()};
	outputs = {delta.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {1}, {0, 0, 0,},					//start index
		{0, 0, -1}, {0}, {0, 0, -1}, 				//negativ end index
		devices, context, bins, q);
	
	//a_tri
	inputs = {zero.data(), delta.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0}, {0,0,0}, {0, 0, 1,},					//start index
		{0}, {0,0,-2}, {0, 0, -1}, 					//negativ end index
		devices, context, bins, q);

	inputs = {a_tri.data(), dzw.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 1}, {1}, {0, 0, 1,},					//start index
		{0, 0, -1}, {-1}, {0, 0, -1}, 				//negativ end index
		devices, context, bins, q);

	inputs = {zero.data(), delta.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0}, {0, 0, Z-2}, {0, 0, Z-1,},				//start index
		{0}, {0, 0, -1}, {0, 0, 0}, 				//negativ end index
		devices, context, bins, q);

	inputs = {a_tri.data(), two.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
		{0, 0, 0}, {0}, {0, 0, 0}, 					//negativ end index
		devices, context, bins, q);

	inputs = {a_tri.data(), dzw.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {Z-1}, {0, 0, Z-1,},			//start index
		{0, 0, 0}, {0}, {0, 0, 0}, 					//negativ end index
		devices, context, bins, q);

	//b_tri
	xt::xarray<double> b_tri_tmp = xt::zeros<double>({X-4, Y-4, Z});
	inputs = {delta.data(), delta.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 1}, {0, 0, 0}, {0, 0, 1,},			//start index
		{0, 0, -1}, {0, 0, -2}, {0, 0, -1},			//negativ end index
		devices, context, bins, q);
	inputs = {b_tri_tmp.data(), dzw.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 1}, {1}, {0, 0, 1,},					//start index
		{0, 0, -1}, {-1}, {0, 0, -1}, 				//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_tmp.data(), one.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, 1}, {0}, {0, 0, 1,},					//start index
		{0, 0, -1}, {0}, {0, 0, -1}, 				//negativ end index
		devices, context, bins, q);
	
	inputs = {sqrttke.data(), mxl.data() };
	outputs = {b_tri_tmp.data()}; //reuse tmp array as we have move intermediate result back to b_tri
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},	//shapes
		{2, 2, 1}, {2, 2, 1}, {0, 0, 1,},			//start index
		{-2, -2, -1}, {-2, -2, -1}, {0, 0, -1},		//negativ end index
		devices, context, bins, q);
	
	xt::xarray<double> c_eps_hack = xt::ones<double>({1}) * c_eps;

	inputs = {b_tri_tmp.data(), c_eps_hack.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, 1}, {0}, {0, 0, 1,},					//start index
		{0, 0, -1}, {0}, {0, 0, -1},				//negativ end index
		devices, context, bins, q);

	inputs = {b_tri.data(), b_tri_tmp.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 1}, {0, 0, 1}, {0, 0, 1,},			//start index
		{0, 0, -1}, {0, 0, -1}, {0, 0, -1},			//negativ end index
		devices, context, bins, q);
	
	//b_tri last index only.
	inputs = {delta.data(), dzw.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-2}, {Z-1}, {0, 0, Z-1},			//start index
		{0, 0, -1}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_tmp.data(), two.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_tmp.data(), one.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {sqrttke.data(), mxl.data() };
	outputs = {b_tri_tmp.data()}; //reuse tmp array as we have move intermediate result back to b_tri
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},		//shapes
		{2, 2, Z-1}, {2, 2, Z-1}, {0, 0, Z-1,},		//start index
		{-2, -2, 0}, {-2, -2, 0}, {0, 0, 0},		//negativ end index
		devices, context, bins, q);
	
	inputs = {b_tri_tmp.data(), c_eps_hack.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri.data(), b_tri_tmp.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, Z-1}, {0, 0, Z-1}, {0, 0, Z-1,},		//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	//b_tri_edge
	inputs = {delta.data(), dzw.data() };
	outputs = {b_tri_edge.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_edge.data(), one.data() };
	outputs = {b_tri_edge.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	xt::xarray<double> b_tri_edge_tmp = xt::zeros_like(b_tri_edge);
	inputs = {sqrttke.data(), mxl.data() };
	outputs = {b_tri_edge_tmp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},//shapes
		{2, 2, 0}, {2, 2, 0}, {0, 0, 0},		//start index
		{-2, -2, 0}, {-2, -2, 0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_edge_tmp.data(), c_eps_hack.data() };
	outputs = {b_tri_edge_tmp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_edge_tmp.data(), b_tri_edge.data() };
	outputs = {b_tri_edge.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},			//negativ end index
		devices, context, bins, q);

	//c_tri
	inputs = {zero.data(), delta.data() };
	outputs = {c_tri.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0}, {0, 0, 0}, {0, 0, 0,},					//start index
		{0}, {0, 0, -1}, {0, 0, -1},				//negativ end index
		devices, context, bins, q);

	inputs = {c_tri.data(), dzw.data() };
	outputs = {c_tri.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0}, {0, 0, 0,},					//start index
		{0, 0, -1}, {-1}, {0, 0, -1},				//negativ end index
		devices, context, bins, q);

	//d_tri	
	inputs = {tke.data(), forc.data() };
	outputs = {d_tri.data()};
	run_broadcast_kernel("add4d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 1}, {X-4, Y-4, Z, 1},	//shapes
		{2, 2, 0, 0}, {2, 2, 0, 0}, {0, 0, 0, 0},		//start index
		{-2, -2, 0, -2}, {-2, -2, 0, 0}, {0, 0, 0, 0},	//negativ end index
		devices, context, bins, q);

	xt::xarray<double> d_tri_tmp = xt::zeros_like(d_tri);
	inputs = {forc_tke_surface.data(), dzw.data() };
	outputs = {d_tri_tmp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, 1}, {Z}, {X-4, Y-4, Z},					//shapes
		{2, 2, 0}, {Z-1}, {0, 0, Z-1},					//start index
		{-2, -2, 0}, {0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {d_tri_tmp.data(), two.data() };
	outputs = {d_tri_tmp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},					//start index
		{0, 0, 0}, {0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	inputs = {d_tri_tmp.data(), d_tri.data() };
	outputs = {d_tri.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, Z-1}, {0, 0, Z-1}, {0, 0, Z-1},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	//SOLVE IMPLICIT FUNCTION UNROLL 
	xt::xarray<double> land_mask = xt::zeros_like(ks);
	xt::xarray<double> edge_mask = xt::zeros_like(a_tri);
	xt::xarray<double> water_mask = xt::zeros_like(a_tri);

	inputs = {ks.data(), zero.data() };
	outputs = {land_mask.data()};
	run_broadcast_kernel("get2d", inputs, outputs, 
		{X-4, Y-4}, {1}, {X-4, Y-4},					//shapes
		{0, 0}, {0}, {0, 0},					//start index
		{0, 0}, {0}, {0, 0},						//negativ end index
		devices, context, bins, q);
	
	xt::xarray<double> Z_dim_arange = xt::arange(Z);
	inputs = {Z_dim_arange.data(), ks.data()};
	outputs = {edge_mask.data()};
	run_broadcast_kernel("eet3d", inputs, outputs, 
		{Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		{0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {land_mask.data(), edge_mask.data()};
	outputs = {edge_mask.data()};
	run_broadcast_kernel("and3d", inputs, outputs, 
		{X-4, Y-4, 1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {Z_dim_arange.data(), ks.data()};
	outputs = {water_mask.data()};
	run_broadcast_kernel("get3d", inputs, outputs, 
		{Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		{0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	std::cout << water_mask << std::endl;

	inputs = {land_mask.data(), water_mask.data()};
	outputs = {water_mask.data()};
	run_broadcast_kernel("and3d", inputs, outputs, 
		{X-4, Y-4, 1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	

	/*
	//apply mask to tridiagonals
	inputs = {a_tri.data(), water_mask.data()};
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	xt::xarray<double> not_edge_mask = xt::zeros_like(edge_mask);

	inputs = {not_edge_mask.data(), zero.data()};
	outputs = {not_edge_mask.data()};
	run_broadcast_kernel("not3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, }, {0, 0, 0},					//start index
		{0, 0, 0}, {0, }, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {not_edge_mask.data(), a_tri.data()};
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0,}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0,}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	*/
	


	std::cout << "sqrttke checksum: should be 1679...: " << xt::sum(sqrttke) << std::endl;
	std::cout << "ks checksum: should be 377...: " << xt::sum(ks) << std::endl;
	std::cout << "delta checksum: should be 85...: " << xt::sum(delta) << std::endl;
	std::cout << "a_tri checksum: should be 689.96...: " << xt::sum(a_tri) << std::endl;
	std::cout << "b_tri checksum: should be -629...: " << xt::sum(b_tri) << std::endl;
	std::cout << "b_tri_edge checksum: should be 527...: " << xt::sum(b_tri_edge) << std::endl;
	std::cout << "c_tri checksum: should be 835...: " << xt::sum(c_tri) << std::endl;
	std::cout << "d_tri checksum: should be 115.9...: " << xt::sum(d_tri) << std::endl;
	std::cout << "land_mask checksum: should be 584...:" << xt::sum(land_mask) << std::endl;
	std::cout << "edge_mask checksum: should be 584...:" << xt::sum(edge_mask) << std::endl;
	std::cout << "water_mask checksum: should be 1759...:" << xt::sum(water_mask) << std::endl;


	return 0;
}