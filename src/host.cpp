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


void pad_begin(std::vector<int> &vec, int val, int to_len){
	int vec_len = vec.size();
	for (int i=0; i<(to_len - vec_len); i++){
		vec.insert(vec.begin(), val);
	}
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

		//calculate size variables
		int dimensions = output_shape.size();
		pad_begin(A_shape, 1, dimensions); //add singleton dimensions
		pad_begin(B_shape, 1, dimensions);
		pad_begin(A_offset, 0, dimensions); //zero pad offsets if not defined
		pad_begin(B_offset, 0, dimensions);
		pad_begin(A_offset_end, 0, dimensions);
		pad_begin(B_offset_end, 0, dimensions);
		for (int i=0; i<dimensions; i++){
			A_offset[i] -= output_offset[i];
			B_offset[i] -= output_offset[i];
		}

		std::vector<int> data_sizes_input = {cumprod(A_shape), cumprod(B_shape)};
		int data_size_output = cumprod(output_shape);


		//calculate strides and offsets
		std::vector<std::vector<int>> input_strides = {get_stride_from_shape(A_shape), get_stride_from_shape(B_shape)};
		std::vector<int> output_stride = get_stride_from_shape(output_shape);
		std::vector<std::vector<int>> input_offset = {A_offset, B_offset};

		//print debug info:
		std::cout << "A_shapes: ";
		print_vec(A_shape);
		std::cout << "B_shapes: ";
		print_vec(B_shape);
		std::cout << "Out_shapes ";
		print_vec(output_shape);

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
			in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * data_sizes_input[i], inputs[i]);
			in_stride_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * dimensions, input_strides[i].data());
			in_offset_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * dimensions, input_offset[i].data());
		}

		out_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double) * data_size_output, outputs[0]);
		out_shape_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(int) * dimensions, output_shape.data());
		out_stride_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * dimensions, output_stride.data());
		out_offset_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * dimensions, output_offset.data());
		out_offset_end_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int) * dimensions, output_offset_end.data());

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

	// Initialization of data

	/* i think this is not needed. idk why i dont get mem errors for using non aligned memory 
	double *u_data = aligned_alloc<double>(size_4d);
	double *v_data = aligned_alloc<double>(size_4d);
	double *w_data = aligned_alloc<double>(size_4d);
	double *maskU_data = aligned_alloc<double>(size_3d);
	double *maskV_data = aligned_alloc<double>(size_3d);
	double *maskW_data = aligned_alloc<double>(size_3d);
	double *dxt_data = aligned_alloc<double>(size_1d_hoz);
	double *dxu_data = aligned_alloc<double>(size_1d_hoz);
	double *dyt_data = aligned_alloc<double>(size_1d_hoz);
	double *dyu_data = aligned_alloc<double>(size_1d_hoz);
	double *dzt_data = aligned_alloc<double>(size_1d_vert);
	double *dzw_data = aligned_alloc<double>(size_1d_vert);
	double *cost_data = aligned_alloc<double>(size_1d_hoz);
	double *cosu_data = aligned_alloc<double>(size_1d_hoz);
	double *kbot_data = aligned_alloc<double>(size_2d);
	double *forc_tke_surface_data = aligned_alloc<double>(size_2d);
	double *kappaM_data = aligned_alloc<double>(size_3d);
	double *mxl_data = aligned_alloc<double>(size_3d);
	double *forc_data = aligned_alloc<double>(size_3d);
	double *tke_data = aligned_alloc<double>(size_4d);
	double *dtke_data = aligned_alloc<double>(size_4d);
	double *flux_east_data = aligned_alloc<double>(size_3d);
	double *flux_north_data = aligned_alloc<double>(size_3d);
	double *flux_top_data = aligned_alloc<double>(size_3d);
	double *temp_data_3d = aligned_alloc<double>(size_3d);
	double *sqrttke_data = aligned_alloc<double>(size_3d);
	double *a_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *b_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *c_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *d_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *delta_data = aligned_alloc<double>((Y-2) * (X - 2));
	*/

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
	xt::xarray<double> a_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> b_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> c_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> d_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> delta = xt::zeros<double>({X-4, Y-4, Z});


	/* this block might be not needed!
	u_data = u.data();
	v_data = v.data();
	w_data = w.data();
	maskU_data = maskU.data();
	maskV_data = maskV.data();
	maskW_data = maskW.data();
	dxt_data = dxt.data();
	dxu_data = dxu.data();
	dyt_data = dyt.data();
	dyu_data = dyu.data();
	dzt_data = dzt.data();
	dzw_data = dzw.data();
	cost_data = cost.data();
	cosu_data = cosu.data();
	kbot_data = kbot.data();
	forc_tke_surface_data = forc_tke_surface.data();
	kappaM_data = kappaM.data();
	mxl_data = mxl.data();
	forc_data = forc.data();
	tke_data = tke.data();
	dtke_data = dtke.data();
	flux_east_data = flux_east.data();
	flux_north_data = flux_north.data();
	flux_top_data = flux_top.data();
	sqrttke_data = sqrttke.data();
	a_tri_data = a_tri.data();
	b_tri_data = b_tri.data();
	c_tri_data = c_tri.data();
	d_tri_data = d_tri.data();
	delta_data = delta.data();
	*/


	int tau = 0;
	double taup1 = 1.;
	double taum1 = 2.;
	double dt_tracer = 1.;
	double dt_mom = 1.;
	double AB_eps = 0.1;
	double alpha_tke = 1.;
	double c_eps = 0.7;
	double K_h_tke = 2000.;

	std::vector<double *> inputs;
	std::vector<double *> outputs;

	/*
	inputs = {
				xt::view(tke, xt::all(), xt::all(), xt::all(), tau).data(), 
				flux_east.data()
			}; 
	outputs = {sqrttke.data()};
	run_kernel("vmax", size_3d, inputs, outputs, devices, context, bins, q);

	inputs = {sqrttke.data()};
	outputs = {sqrttke.data()};
	run_kernel("vsqrt", size_3d, inputs, outputs, devices, context, bins, q);
	*/

	xt::xarray<double> one = xt::ones<double>({1});
	xt::xarray<double> test = xt::arange(9).reshape({3,3});
	xt::xarray<double> res = xt::zeros<double>({4, 4});

	inputs = {	
				one.data(),
				dzt.data(),
			};
	outputs = { delta.data() };
	run_broadcast_kernel("div3d", inputs, outputs, 
		{1}, {Z}, {X-4, Y-4, Z},		//shapes
		{0}, {1,}, {0, 0, 0,},			//start index
		{0}, {0,}, {0, 0, -1}, 		//negativ end index
		devices, context, bins, q);
	std::cout << delta;

	return 0;
}
