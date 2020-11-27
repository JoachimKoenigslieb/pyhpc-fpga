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
	if (vec.size() > to_len){
		for (int i=0; i<(vec.size() - to_len) +1; i++){
			vec.pop_back();
		}
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

std::vector<int> squeeze(std::vector<int> A){
	//squeezing removes all ones! (singleton dimensions)
	std::vector<int> squeezed;
	for (int i=0; i<A.size(); i++){
		if (A[i] != 1){
			squeezed.push_back(A[i]);
		}
	}
	return squeezed;
}

std::vector<int> broadcast(std::vector<int> A, std::vector<int> B){
	int max_dim = std::max(A.size(), B.size());
	pad_begin(A, 1, max_dim);
	pad_begin(B, 1, max_dim);
	
	std::vector<int> broadcasted_shape;
	for (int i=0; i<max_dim; i++){
		if (A[i] == 1 && B[i] == 1 && A[i] == B[i]){
			broadcasted_shape[i] = std::max(A[i], B[i]);
		} else {
			std::cout << "SHAPES DO NOT BROADCAST!";
			exit(1); //break (probably in a not-responsible way??? idk)
		}
	}

	return broadcasted_shape;
}

void negotiate_shapes(std::vector<int> out_view_shape,std::vector<int> A_view_shape, std::vector<int> B_view_shape){
	std::vector<int> A_view_shape_squeeze, B_view_shape_squeeze;
	A_view_shape_squeeze = squeeze(A_view_shape);
	B_view_shape_squeeze = squeeze(B_view_shape);

	std::vector<int> broadcasted_shape = broadcast(A_view_shape_squeeze, B_view_shape_squeeze);
	int dim = std::max(broadcasted_shape.size(), out_view_shape.size());
	for (int i=0; i<dim; i++){
		if (broadcasted_shape[i] != out_view_shape[i]){
			std::cout << "SHAPES DO NOT BROADCAST! (doesnt fit into output)";
			exit(1);
		}
	}
}

std::vector<int> stride_from_shape(std::vector<int> A){
	int dims = A.size();
	std::vector<int> stride(dims);
	stride[dims - 1] = 1;
	for (int i=dims-2; i>-1; i--){
		stride[i] = stride[i + 1] * A[i + 1];
	}

	return stride;
}

std::vector<int> filter_on_squeeze(std::vector<int> view_shape, std::vector<int> A){
	assert(view_shape.size() == A.size()); //should be equal!

	std::vector<int> new_A;
	int size = view_shape.size();

	for (int i=0; i<size; i++){
		if (view_shape[i] != 1){
			new_A.push_back(A[i]);
		}
	}

	return new_A;
}

std::vector<int> zero_on_squeeze(std::vector<int> view_shape, std::vector<int> A){
	assert(view_shape.size() == A.size()); //should be equal!

	std::vector<int> new_A;
	int size = view_shape.size();

	for (int i=0; i<size; i++){
		if (view_shape[i] != 1){
			new_A.push_back(A[i]);
		} else {
			new_A.push_back(0);
		}
	}

	return new_A;
}


int collect_linear_offset(std::vector<int> view_shape, std::vector<int> stride, std::vector<int> offset){
	int dims = view_shape.size();
	int lin_offset=0;

	for (int i=0; i<dims; i++){
		if (view_shape[i] == 1)
			lin_offset += stride[i] * offset[i];
	}
	
	return lin_offset;
}

std::vector<int> rebuild_stride(std::vector<int> stride, std::vector<int> view_shape, int out_dim){
	int dims = stride.size();
	std::vector<int> new_stride;

	new_stride = zero_on_squeeze(view_shape, stride);
	pad_begin(new_stride, 0, out_dim);

	return new_stride;
}

std::vector<int> rebuild_offset(std::vector<int> offset, std::vector<int> view_shape, std::vector<int> out_offset, int out_dim){
	int dims = view_shape.size();
	std::vector<int> filtered_offset;

	filtered_offset = zero_on_squeeze(view_shape, offset);

	pad_begin(filtered_offset, 0, out_dim);

	std::vector<int> new_offset = sub_vecs(filtered_offset, out_offset);
	return new_offset;
}


void negotiate_strides(	std::vector<int> A_shape, std::vector<int> &out_shape, 
					std::vector<int> &A_offset, std::vector<int> &out_offset,
					std::vector<int> A_end_offset, std::vector<int> &out_end_offset,
					std::vector<int> &A_stride_res, std::vector<int> &out_stride_res,
					int &A_lin_offset_res, int &out_lin_offset_res, int &out_dim,
					int &A_data_size, int &out_data_size
					){						
	std::vector<int> A_view_shape, B_view_shape, out_view_shape;
	A_view_shape = sub_vecs(add_vecs(A_shape, A_end_offset), A_offset); //A_end_offset is a negative index indicating how many to take from the end. 
	out_view_shape = sub_vecs(add_vecs(out_shape, out_end_offset), out_offset);

	//negotiate_shapes(out_view_shape, A_view_shape, B_view_shape); //This checks if we can broadcast at all

	std::vector<int> A_stride, B_stride, out_stride;
	A_stride = stride_from_shape(A_shape);
	out_stride = stride_from_shape(out_shape);

	int A_lin_offset, B_lin_offset, out_lin_offset;
	A_lin_offset = collect_linear_offset(A_view_shape, A_stride, A_offset);
	out_lin_offset = collect_linear_offset(out_view_shape, out_stride, out_offset);

	out_dim = squeeze(out_view_shape).size(); //should be squeezed!

	A_stride = rebuild_stride(A_stride, A_view_shape, out_dim);
	out_stride = rebuild_stride(out_stride, out_view_shape, out_dim);

	A_offset = rebuild_offset(A_offset, A_view_shape, out_offset, out_dim);

	out_offset = filter_on_squeeze(out_view_shape, out_offset);
	out_end_offset = filter_on_squeeze(out_view_shape, out_end_offset);

	A_data_size = cumprod(A_shape);
	out_data_size = cumprod(out_shape);

	out_shape = filter_on_squeeze(out_view_shape, out_shape); 

	A_stride_res = A_stride;
	out_stride_res = out_stride;

	A_lin_offset_res = A_lin_offset;
	out_lin_offset_res = out_lin_offset;
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

		int num_in = 2;
		int dimensions;

		//setup buffers:
		std::vector<cl::Buffer> in_buffers(num_in);
		std::vector<cl::Buffer> in_stride_buffers(num_in);
		std::vector<cl::Buffer> in_offset_buffers(num_in);

		cl::Buffer out_buffer;
		cl::Buffer out_shape_buffer;
		cl::Buffer out_stride_buffer;
		cl::Buffer out_offset_buffer;
		cl::Buffer out_offset_end_buffer;

		std::vector<int> A_stride, B_stride, output_stride;
		int A_lin_offset, B_lin_offset, output_lin_offset;
		int output_data_size, B_data_size, A_data_size;
	
		std::vector<int> output_offset_B_copy(output_offset);
		std::vector<int> output_offset_end_B_copy(output_offset_end);
		std::vector<int> output_stride_B_copy(output_stride);
		std::vector<int> output_shape_B_copy(output_shape);
		int output_lin_offset_B_copy, output_data_size_B_copy;

		negotiate_strides(A_shape, output_shape, A_offset, output_offset, A_offset_end, output_offset_end, A_stride, output_stride, A_lin_offset, output_lin_offset, dimensions, A_data_size, output_data_size);
		negotiate_strides(B_shape, output_shape_B_copy, B_offset, output_offset_B_copy, B_offset_end, output_offset_end_B_copy, B_stride, output_stride_B_copy, B_lin_offset, output_lin_offset_B_copy, dimensions, B_data_size, output_data_size_B_copy);

		//create "looping over" vectors (seems stupid to it this way...)
		std::vector<std::vector<int>> input_strides = {A_stride, B_stride};
		std::vector<std::vector<int>> input_offset = {A_offset, B_offset};
		std::vector<int> data_sizes_input = {A_data_size, B_data_size};

		//print debug info:
		std::cout << "Sending n="<<dimensions << std::endl;

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

		std::cout << "A_lin_offset: " << A_lin_offset << std::endl;
		std::cout << "B_lin_offset: " << B_lin_offset << std::endl;
		std::cout << "output_lin_offset: " << output_lin_offset << std::endl;

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
		std::cout << "Output size: " << output_data_size << std::endl;

		// write to buffers
		for (int i = 0; i < num_in; i++){
			in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * data_sizes_input[i], inputs[i]);
			in_stride_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_strides[i].data());
			in_offset_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_offset[i].data());
			assert(dimensions == input_strides[i].size());
			assert(dimensions == input_offset[i].size());
		}

		out_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * output_data_size, outputs[0]);
		out_shape_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_shape.data());
		out_stride_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_stride.data());
		out_offset_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset.data());
		out_offset_end_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset_end.data());

		assert(output_shape.size() == dimensions);
		assert(output_stride.size() == dimensions);
		assert(output_offset.size() == dimensions);
		assert(output_offset_end.size() == dimensions);

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

		std::cout << "INFO: offsets has been set." <<std::endl;

		kernel.setArg(6, A_lin_offset);
		kernel.setArg(7, B_lin_offset);
		kernel.setArg(13, output_lin_offset); 

		std::cout << "INFO: linear offsets has been set." <<std::endl;

		//set outputs
		kernel.setArg(4 * num_in, out_buffer);			  //arg8
		kernel.setArg(4 * num_in + 1, out_shape_buffer);	  //arg9
		kernel.setArg(4 * num_in + 2, out_stride_buffer); //arg10
		kernel.setArg(4 * num_in + 3, out_offset_buffer);		  //arg11
		kernel.setArg(4 * num_in + 4, out_offset_end_buffer);  //arg12


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

void run_gtsv(std::string kernel_name, int kernel_size, std::vector<double *> &inputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
        // this is a helper function to execute a kernel.
        {
                cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
                cl::Kernel kernel(program, kernel_name.data());

                 // DDR Settings
		std::vector<cl_mem_ext_ptr_t> mext_io(4);
		mext_io[0].flags = XCL_MEM_DDR_BANK0;
		mext_io[1].flags = XCL_MEM_DDR_BANK0;
		mext_io[2].flags = XCL_MEM_DDR_BANK0;
		mext_io[3].flags = XCL_MEM_DDR_BANK0;

		mext_io[0].obj = inputs[0];
		mext_io[0].param = 0;
		mext_io[1].obj = inputs[1];
		mext_io[1].param = 0;
		mext_io[2].obj = inputs[2];
		mext_io[2].param = 0;
		mext_io[3].obj = inputs[3];
		mext_io[3].param = 0;

		// Create device buffer and map dev buf to host buf
		cl::Buffer matdiaglow_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
										sizeof(double) * kernel_size, &mext_io[0]);
		cl::Buffer matdiag_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
									sizeof(double) * kernel_size, &mext_io[1]);
		cl::Buffer matdiagup_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
										sizeof(double) * kernel_size, &mext_io[2]);
		cl::Buffer rhs_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
								sizeof(double) * kernel_size, &mext_io[3]);

		// Data transfer from host buffer to device buffer
		std::vector<std::vector<cl::Event> > kernel_evt(2);
		kernel_evt[0].resize(1);
		kernel_evt[1].resize(1);

		std::vector<cl::Memory> ob_in, ob_out;
		ob_in.push_back(matdiaglow_buffer);
		ob_in.push_back(matdiag_buffer);
		ob_in.push_back(matdiagup_buffer);
		ob_in.push_back(rhs_buffer);

		ob_out.push_back(rhs_buffer);

		q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &kernel_evt[0][0]); // 0 : migrate from host to dev
		q.finish();
		std::cout << "INFO: Finish data transfer from host to device" << std::endl;


		// Setup kernel
		kernel.setArg(0, kernel_size);
		kernel.setArg(1, matdiaglow_buffer);
		kernel.setArg(2, matdiag_buffer);
		kernel.setArg(3, matdiagup_buffer);
		kernel.setArg(4, rhs_buffer);
		q.finish();
		std::cout << "INFO: Finish kernel setup" << std::endl;

		q.enqueueTask(kernel, nullptr, nullptr);
		q.finish();
		q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr); // 1 : migrate from dev to host
		q.finish();


		}
}

void run_where_kernel(std::string kernel_name, 
							std::vector<double *> &inputs,
							std::vector<double *> &outputs,
							std::vector<int> A_shape,
							std::vector<int> B_shape,
							std::vector<int> C_shape,
							std::vector<int> output_shape,
							std::vector<int> A_offset,
							std::vector<int> B_offset,
							std::vector<int> C_offset,
							std::vector<int> output_offset,
							std::vector<int> A_offset_end,
							std::vector<int> B_offset_end,
							std::vector<int> C_offset_end,
							std::vector<int> output_offset_end,
							std::vector<cl::Device> &devices,
							cl::Context &context,
							cl::Program::Binaries &bins,
							cl::CommandQueue &q)
{
	{
		cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
		cl::Kernel kernel(program, kernel_name.data());
		std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

		//num inputs outputs
		int num_in = inputs.size();
		int dimensions;
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

		std::vector<int> A_stride, B_stride, C_stride, output_stride;
		int A_lin_offset, B_lin_offset, C_lin_offset, output_lin_offset;
		int output_data_size, C_data_size, B_data_size, A_data_size;
	
		std::vector<int> output_offset_B_copy(output_offset);
		std::vector<int> output_offset_end_B_copy(output_offset_end);
		std::vector<int> output_stride_B_copy(output_stride);
		std::vector<int> output_shape_B_copy(output_shape);
		int output_lin_offset_B_copy, output_data_size_B_copy;

		std::vector<int> output_offset_C_copy(output_offset);
		std::vector<int> output_offset_end_C_copy(output_offset_end);
		std::vector<int> output_stride_C_copy(output_stride);
		std::vector<int> output_shape_C_copy(output_shape);
		int output_lin_offset_C_copy, output_data_size_C_copy;


		negotiate_strides(A_shape, output_shape, A_offset, output_offset, A_offset_end, output_offset_end, A_stride, output_stride, A_lin_offset, output_lin_offset, dimensions, A_data_size, output_data_size);
		negotiate_strides(B_shape, output_shape_B_copy, B_offset, output_offset_B_copy, B_offset_end, output_offset_end_B_copy, B_stride, output_stride_B_copy, B_lin_offset, output_lin_offset_B_copy, dimensions, B_data_size, output_data_size_B_copy);
		negotiate_strides(C_shape, output_shape_C_copy, C_offset, output_offset_C_copy, C_offset_end, output_offset_end_C_copy, C_stride, output_stride_C_copy, C_lin_offset, output_lin_offset_C_copy, dimensions, C_data_size, output_data_size_C_copy);


		//create "looping over" vectors (seems stupid to it this way...)
		std::vector<std::vector<int>> input_strides = {A_stride, B_stride, C_stride};
		std::vector<std::vector<int>> input_offset = {A_offset, B_offset, C_offset};
		std::vector<int> data_sizes_input = {A_data_size, B_data_size, C_data_size};

		//print debug info:
		std::cout << "A_shapes: ";
		print_vec(A_shape);
		std::cout << "B_shapes: ";
		print_vec(B_shape);
		std::cout << "C_shapes: ";
		print_vec(C_shape);
		std::cout << "Out_shapes ";
		print_vec(output_shape);

		std::cout << "A_offset: ";
		print_vec(A_offset);
		std::cout << "B_offset: ";
		print_vec(B_offset);
		std::cout << "C_offset: ";
		print_vec(C_offset);
		std::cout << "out_offset: ";
		print_vec(output_offset);


		std::cout << "A_offset_end: ";
		print_vec(A_offset_end);		
		std::cout << "B_offset_end: ";
		print_vec(B_offset_end);
		std::cout << "C_offset_end: ";
		print_vec(C_offset_end);		
		std::cout << "out_offset_end: ";
		print_vec(output_offset_end);

		std::cout << "A strides: ";
		print_vec(input_strides[0]);
		std::cout << "B strides: ";
		print_vec(input_strides[1]);
		std::cout << "C strides: ";
		print_vec(input_strides[2]);
		std::cout << "O strides: ";
		print_vec(output_stride);

		std::cout << "Input size: " << data_sizes_input[0] << ", " << data_sizes_input[1] << ", " << data_sizes_input[2] << std::endl;
		std::cout << "Output size: " << output_data_size << std::endl;

		// write to buffers
		for (int i = 0; i < num_in; i++){
			in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * data_sizes_input[i], inputs[i]);
			in_stride_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_strides[i].data());
			in_offset_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, input_offset[i].data());
			assert(dimensions == input_strides[i].size());
			assert(dimensions == input_offset[i].size());
		}

		out_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * output_data_size, outputs[0]);
		out_shape_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_shape.data());
		out_stride_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_stride.data());
		out_offset_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset.data());
		out_offset_end_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * dimensions, output_offset_end.data());

		assert(output_shape.size() == dimensions);
		assert(output_stride.size() == dimensions);
		assert(output_offset.size() == dimensions);
		assert(output_offset_end.size() == dimensions);

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

		kernel.setArg(9, A_lin_offset);
		kernel.setArg(10, B_lin_offset);
		kernel.setArg(11, C_lin_offset); 
		kernel.setArg(17, output_lin_offset); 

		std::cout << "INFO: linear offsets has been set." <<std::endl;

		//set outputs
		kernel.setArg(4 * num_in, out_buffer);			  //arg6
		kernel.setArg(4 * num_in + 1, out_shape_buffer);	  //arg7
		kernel.setArg(4 * num_in + 2, out_stride_buffer); //arg8
		kernel.setArg(4 * num_in + 3, out_offset_buffer);		  //arg9
		kernel.setArg(4 * num_in + 4, out_offset_end_buffer);  //arg10

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

void adv_superbee(xt::xarray<double> &vel, xt::xarray<double> &var, xt::xarray<double> &mask, xt::xarray<double> &dx, int axis, xt::xarray<double> &cost, xt::xarray<double> &cosu,
							std::vector<cl::Device> &devices,
							cl::Context &context,
							cl::Program::Binaries &bins,
							cl::CommandQueue &q){

	// start by getting indexing:
	std::vector<std::vector<int>> starts, ends, mask_starts, mask_ends;
	std::vector<double*> inputs, outputs;

	// permute depending on which axis we are working

	xt::xarray<double> zero = xt::zeros<double>({1});
	xt::xarray<double> velfac, dx_local;
	std::vector<int> dx_local_shape, velfac_shape, velfac_starts, velfac_ends, dx_shape;

	if (axis==0){
		for (int n=-1; n<3; n++){
			starts.push_back({1 + n, 2, 0, 0});
			ends.push_back({-2 + n, -2 , 0, -2});
			mask_starts.push_back({1 + n, 2, 0, });
			mask_ends.push_back({-2 + n, -2 , 0, }); // the masks are 3d sized {X, Y, Z}, we can't have the last (constnat) index on them. 
		}
			dx_local_shape = {X-3, Y-4, 1};
			dx_shape = {X};
			velfac_shape = {1};
			velfac_starts = {0,};
			velfac_ends = {0,};

			dx_local = xt::zeros<double>(dx_local_shape);
			velfac = xt::ones<double>({1});

			inputs = {cost.data(), dx.data()};
			outputs = {dx_local.data()};
			run_broadcast_kernel("mult2d", inputs, outputs, 
				{1, Y, 1}, {X, 1 ,1 }, dx_local_shape,					//shapes
				{0, 2, 0}, {1, 0, 0}, {0, 0, 0},						//start index
				{0, -2, 0}, {-2, 0, 0,}, {0, 0, 0},					//negativ end index
				devices, context, bins, q);	
	} 
	/*if (axis==1){
		for (int n=-1; n<3; n++){
			starts.push_back({2, 1+n, 0, 1});
			ends.push_back({-2, -2+n, 0, -1});
		}

		std::vector<int> dx_local_shape = {1, 29, 1};
		std::vector<int> velfac_shape = {1, 29, 1};
		std::vector<int> velfac_starts = {0, 0, 0};
		std::vector<int> velfac_ends = {0, 0, 0};

		xt::xarray<double> dx_local = xt::zeros<double>({dx_local_shape});

		inputs = {cost.data(), dx.data()};
		outputs = {dx_local.data()};
		run_broadcast_kernel("mult1d", inputs, outputs, 
			{Y}, {X,}, dx_local_shape,					//shapes
			{1}, {1,}, {0, 0, 0},						//start index
			{-2}, {-2,}, {0, 0, 0},					//negativ end index
			devices, context, bins, q);	

		inputs = {zero.data(), cosu.data()};
		outputs = {dx_local.data()};
		run_broadcast_kernel("add1d", inputs, outputs, 
			{0}, {X,}, velfac_shape,					//shapes
			{0}, {1,}, {0, 0, 0},						//start index
			{0}, {-2,}, {0, 0, 0},					//negativ end index
			devices, context, bins, q);	
	} 
	if (axis==2){
		for (int n=-1; n<3; n++){
			starts.push_back({2, 2, 1+n, 1});
			ends.push_back({-2, -2, -2+n, -1});
		}
			std::vector<int> dx_local_shape = {1, 1, 3};
			std::vector<int> velfac_shape = {1,};
			std::vector<int> velfac_starts = {0,};
			std::vector<int> velfac_ends = {0,};

			xt::xarray<double> velfac = xt::ones<double>({1});
	} */

	//start contain start index of the slices. in pyhpc benchark they are called sm1, s, sp1, sp2. starting index of s is then starts[1]

	std::vector<int> intermediate_shape = {X-starts[1][0] + ends[1][0], Y - starts[1][1] + ends[1][1], Z - starts[1][2] + ends[1][2]};
	xt::xarray<double> uCFL = xt::zeros<double>(intermediate_shape);
	xt::xarray<double> rjp = xt::zeros<double>(intermediate_shape);
	xt::xarray<double> rj = xt::zeros<double>(intermediate_shape);
	xt::xarray<double> rjm = xt::zeros<double>(intermediate_shape);
	xt::xarray<double> cr = xt::zeros<double>(intermediate_shape);

	//ucfl
	inputs = {velfac.data(), vel.data()};
	outputs = {uCFL.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		velfac_shape, {X, Y, Z, 3}, intermediate_shape,					//shapes
		velfac_starts, starts[1], {0, 0, 0},						//start index
		velfac_ends, ends[1], {0, 0, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {uCFL.data(), dx_local.data()};
	outputs = {uCFL.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		intermediate_shape, dx_local_shape, intermediate_shape,					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {uCFL.data(), zero.data()}; //second input does nothing abs only takes one input. needs to be here because run broadcast kernel is shit
	outputs = {uCFL.data()};
	run_broadcast_kernel("abs3d", inputs, outputs, 
		intermediate_shape, {1}, intermediate_shape,					//shapes
		{0, 0, 0}, {0}, {0, 0, 0},						//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	// rjp
	inputs = {vel.data(), vel.data()};
	outputs = {rjp.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 3}, intermediate_shape,					//shapes
		start[3], start[2], {0, 0, 0},						//start index
		end[3], end[2], {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {mask.data(), rjp.data()};
	outputs = {rjp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z,}, intermediate_shape, intermediate_shape,					//shapes
		mask_start[2], {0, 0, 0}, {0, 0, 0},						//start index
		mask_end[2], {0, 0, 0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	//rj
	inputs = {vel.data(), vel.data()};
	outputs = {rj.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 3}, intermediate_shape,					//shapes
		start[2], start[1], {0, 0, 0},						//start index
		end[2], end[1], {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {mask.data(), rjp.data()};
	outputs = {rj.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z,}, intermediate_shape, intermediate_shape,					//shapes
		mask_start[1], {0, 0, 0}, {0, 0, 0},						//start index
		mask_end[1], {0, 0, 0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	//rjm
	inputs = {vel.data(), vel.data()};
	outputs = {rjm.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 3}, intermediate_shape,					//shapes
		start[1], start[0], {0, 0, 0},						//start index
		end[1], end[0], {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {mask.data(), rjmp.data()};
	outputs = {rjm.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z,}, intermediate_shape, intermediate_shape,					//shapes
		mask_start[0], {0, 0, 0}, {0, 0, 0},						//start index
		mask_end[0], {0, 0, 0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	//unroll cr. some confusing stuff is in here!
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
	std::string xclbin_path = "./sw_emu_kernels.xclbin";
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
	xt::xarray<double> tke_surf_corr = xt::zeros<double>({X,Y});


	int tau = 0;
	double taup1 = 1.;
	double taum1 = 2.;
	double dt_tracer = 1.;
	double dt_mom = 1.;
	double dt_tke = 1.;
	double AB_eps = 0.1;
	double alpha_tke = 1.;
	double c_eps = 0.7;

	std::vector<double *> inputs;
	std::vector<double *> outputs;

	int tmp_int;

	xt::xarray<double> one = xt::ones<double>({1});
	xt::xarray<double> zero = xt::zeros<double>({1});
	xt::xarray<double> half = xt::ones<double>({1}) * 0.5;
	xt::xarray<double> two = xt::ones<double>({1}) * 2;
	xt::xarray<double> K_h_tke = xt::ones<double>({1}) * 2000;

	//kbot
	inputs = {kbot.data(), one.data()};
	outputs = {ks.data()};
	run_broadcast_kernel("sub2d", inputs, outputs, 
		{X, Y}, {1}, {X-4, Y-4,},			//shapes
		{2, 2}, {0}, {0, 0},			//start index
		{-2, -2}, {0}, {0, 0}, 			//negativ end index
		devices, context, bins, q);

	//sqrttke
	inputs = {tke.data(), zero.data()};
	outputs = {sqrttke.data()};
	run_broadcast_kernel("max3d", inputs, outputs, 
		{X, Y, Z, 3}, {1}, {X, Y, Z,},			//shapes
		{0, 0, 0, 0}, {0}, {0, 0, 0,},			//start index
		{0, 0, 0, -2}, {0}, {0, 0, 0}, 			//negativ end index
		devices, context, bins, q);

	inputs = {sqrttke.data(), zero.data()}; //sqrt takes one argument. zero does nothing
	outputs = {sqrttke.data()};
	run_broadcast_kernel("sqrt3d", inputs, outputs, 
		{X, Y, Z,}, {1}, {X, Y, Z,},			//shapes
		{0, 0, 0,}, {0}, {0, 0, 0,},			//start index
		{0, 0, 0,}, {0}, {0, 0, 0}, 			//negativ end index
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
	run_broadcast_kernel("sub2d", inputs, outputs, 
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0}, {0, 0, Z-2}, {0, 0, Z-1,},				//start index
		{0}, {0, 0, -1}, {0, 0, 0}, 				//negativ end index
		devices, context, bins, q);

	inputs = {a_tri.data(), two.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
		{0, 0, 0}, {0}, {0, 0, 0}, 					//negativ end index
		devices, context, bins, q);

	inputs = {a_tri.data(), dzw.data(), };
	outputs = {a_tri.data()};
	run_broadcast_kernel("div2d", inputs, outputs, 
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
	run_broadcast_kernel("div2d", inputs, outputs, 
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-2}, {Z-1}, {0, 0, Z-1},			//start index
		{0, 0, -1}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_tmp.data(), two.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri_tmp.data(), one.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {sqrttke.data(), mxl.data() };
	outputs = {b_tri_tmp.data()}; //reuse tmp array as we have move intermediate result back to b_tri
	run_broadcast_kernel("div2d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},		//shapes
		{2, 2, Z-1}, {2, 2, Z-1}, {0, 0, Z-1,},		//start index
		{-2, -2, 0}, {-2, -2, 0}, {0, 0, 0},		//negativ end index
		devices, context, bins, q);
	
	inputs = {b_tri_tmp.data(), c_eps_hack.data() };
	outputs = {b_tri_tmp.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {b_tri.data(), b_tri_tmp.data() };
	outputs = {b_tri.data()};
	run_broadcast_kernel("add2d", inputs, outputs, 
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
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z,}, {X-4, Y-4, Z, },	//shapes
		{2, 2, 0, 0}, {2, 2, 0,}, {0, 0, 0,},		//start index
		{-2, -2, 0, -2}, {-2, -2, 0,}, {0, 0, 0,},	//negativ end index
		devices, context, bins, q);

	xt::xarray<double> d_tri_tmp = xt::zeros_like(d_tri);
	std::cout << "d_tri tmp 0 (should be zero!)" << xt::sum(d_tri_tmp) << std::endl;

	xt::xarray<double> test_tmp = xt::zeros<double>({28, 28});


	inputs = {forc_tke_surface.data(), dzw.data() };
	outputs = {d_tri_tmp.data()};
	run_broadcast_kernel("div2d", inputs, outputs, 
		{X, Y,}, {Z}, {X-4, Y-4,Z} ,					//shapes
		{2, 2,}, {Z-1}, {0, 0, Z-1},					//start index
		{-2, -2,}, {0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	std::cout << "d_tri tmp 1 (should be 27)" << xt::sum(d_tri_tmp) << std::endl;
	
	inputs = {d_tri_tmp.data(), two.data() };
	outputs = {d_tri_tmp.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},					//start index
		{0, 0, 0}, {0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	inputs = {d_tri_tmp.data(), d_tri.data() };
	outputs = {d_tri.data()};
	run_broadcast_kernel("add2d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
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
		{1, 1, Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
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
		{1, 1, Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	inputs = {land_mask.data(), water_mask.data()};
	outputs = {water_mask.data()};
	run_broadcast_kernel("and3d", inputs, outputs, 
		{X-4, Y-4, 1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	//apply mask to tridiagonals
	inputs = {a_tri.data(), water_mask.data()};
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	xt::xarray<double> not_edge_mask = xt::zeros_like(edge_mask);
	
	inputs = {edge_mask.data(), zero.data()}; //not op actually takes 1 operand. the zero in this case is noop and be anything!
	outputs = {not_edge_mask.data()};
	run_broadcast_kernel("not3d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		{0, 0, 0}, {0, }, {0, 0, 0},					//start index
		{0, 0, 0}, {0, }, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {not_edge_mask.data(), a_tri.data()};
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);
	
	inputs = {water_mask.data(), b_tri.data(), one.data()};
	outputs = {b_tri.data()};
	run_where_kernel("where3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},		//shapes
		{0, 0, 0}, {0, 0, 0}, {0,}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0,}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {edge_mask.data(), b_tri_edge.data(), b_tri.data()};
	outputs = {b_tri.data()};
	run_where_kernel("where3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4,Y-4,Z}, {X-4, Y-4, Z},		//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {c_tri.data(), water_mask.data()};
	outputs = {c_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	inputs = {d_tri.data(), water_mask.data()};
	outputs = {d_tri.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);


	//prepare inputs for gtsv
	inputs = {a_tri.data(), zero.data()}; //This is such a hack.. We need to write and assignment kernel also!
	outputs = {a_tri.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0,}, {0, 0, 0},						//start index
		{0, 0, -Z+1}, {0,}, {0, 0, -Z+1},						//negativ end index
		devices, context, bins, q);

	inputs = {c_tri.data(), zero.data()}; //This is such a hack.. We need to write and assignment kernel also!
	outputs = {c_tri.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0,}, {0, 0, Z-1},						//start index
		{0, 0, 0}, {0,}, {0, 0, 0},						//negativ end index
		devices, context, bins, q);

	a_tri[0] = 0; 	//on a tri-diagonal matrix, upper and lower diagonals have n-1 entries. We zero them like this to fit gtsv kernel convetion
	c_tri[(X-4)*(Y-4)*Z - 1] = 0;

	
	inputs = {a_tri.data(), b_tri.data(), c_tri.data(), d_tri.data()};
	run_gtsv("gtsv", (X-4)*(Y-4)*Z, inputs, devices, context, bins, q); //this outputs ans into d_tri (xilinx solver kernel choice, not mine)

	
	inputs = {water_mask.data(), d_tri.data(), tke.data()};
	outputs = {tke.data()};
	run_where_kernel("where3d", inputs, outputs,
	{X-4, Y-4, Z,}, {X-4, Y-4, Z,}, {X, Y, Z, 3}, {X, Y, Z, 3},
	{0, 0, 0}, {0, 0, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},
	{0, 0, 0}, {0, 0, 0}, {-2, -2, 0, -1}, {-2, -2, 0, -1},
	devices, context, bins, q);

	xt::xarray<double> mask = xt::zeros<double>({X-4,Y-4});
	
	inputs = {zero.data(), tke.data()};
	outputs = {mask.data()};
	run_broadcast_kernel("gt2d", inputs, outputs, 
		{1}, {X, Y, Z, 3}, {X-4, Y-4,},					//shapes
		{0}, {2, 2, Z-1, 1}, {0, 0,},						//start index
		{0}, {-2, -2, 0, -1}, {0, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {zero.data(), tke.data()};
	outputs = {tke_surf_corr.data()};
	run_broadcast_kernel("sub2d", inputs, outputs, 
		{1}, {X, Y, Z, 3}, {X, Y,},					//shapes
		{0}, {2, 2, Z-1, 1}, {2, 2,},						//start index
		{0}, {-2, -2, 0, -1}, {-2, -2, },					//negativ end index
		devices, context, bins, q);	

	
	inputs = {half.data(), tke_surf_corr.data()};
	outputs = {tke_surf_corr.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{1}, {X, Y, }, {X, Y,},					//shapes
		{0}, {2, 2,}, {2, 2,},						//start index
		{0}, {-2, -2}, {-2, -2, },					//negativ end index
		devices, context, bins, q);	

	inputs = {dzw.data(), tke_surf_corr.data()};
	outputs = {tke_surf_corr.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{Z}, {X, Y, }, {X, Y,},					//shapes
		{Z-1}, {2, 2,}, {2, 2,},						//start index
		{0}, {-2, -2}, {-2, -2, },					//negativ end index
		devices, context, bins, q);	
	
	inputs = {mask.data(), tke_surf_corr.data(), zero.data()};
	outputs = {tke_surf_corr.data()};
	run_where_kernel("where2d", inputs, outputs,
	{X-4, Y-4,}, {X, Y}, {1}, {X, Y,},
	{0, 0,}, {2, 2,}, {0,}, {2, 2,},
	{0, 0,}, {-2,-2,}, {0,}, {-2, -2,},
	devices, context, bins, q);

	// flux east
	inputs = {tke.data(), tke.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z},					//shapes
		{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,},						//start index
		{0, 0, 0, -2}, {-1, 0, 0, -2}, {-1, 0, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {K_h_tke.data(), flux_east.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{1}, {X, Y, Z,}, {X, Y, Z},					//shapes
		{0,}, {0, 0, 0,}, {0, 0, 0,},						//start index
		{0,}, {-1, 0, 0, }, {-1, 0, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_east.data(), cost.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {1, X, 1}, {X, Y, Z},					//shapes
		{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		{-1, 0, 0}, { 0, 0, 0}, {-1, 0, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_east.data(), dxu.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {X, 1, 1}, {X, Y, Z},					//shapes
		{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		{-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0, },					//negativ end index
		devices, context, bins, q);
	
	inputs = {flux_east.data(), maskU.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		{-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_east.data(), zero.data()};
	outputs = {flux_east.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X, Y, Z}, {1}, {X, Y, Z},					//shapes
		{X-1, 0, 0,}, {0,}, {X-1, 0, 0,},						//start index
		{0, 0, 0}, {0,}, {0, 0, 0, },					//negativ end index
		devices, context, bins, q);

	//flux norht
	inputs = {tke.data(), tke.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z},					//shapes
		{0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,},						//start index
		{0, 0, 0, -2}, {0, -1, 0, -2}, {0, -1, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {K_h_tke.data(), flux_north.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{1}, {X, Y, Z,}, {X, Y, Z},					//shapes
		{0,}, {0, 0, 0,}, {0, 0, 0,},						//start index
		{0,}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_north.data(), dyu.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0,},						//start index
		{0, -1, 0}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_north.data(), cosu.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0,},						//start index
		{0, -1, 0}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_north.data(), maskV.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0,},						//start index
		{0, -1, 0}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		devices, context, bins, q);

	inputs = {flux_north.data(), zero.data()};
	outputs = {flux_north.data()};
	run_broadcast_kernel("mult2d", inputs, outputs, 
		{X, Y, Z}, {1}, {X, Y, Z},					//shapes
		{0, Y-1, 0,}, {0,}, {0, Y-1, 0,},						//start index
		{0, 0, 0}, {0,}, {0, 0, 0, },					//negativ end index
		devices, context, bins, q);

	// Tke temp stuff
	xt::xarray<double> tke_temp = xt::zeros<double>({X, Y, Z});

	inputs = {flux_east.data(), flux_east.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{2, 2, 0,}, {1, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {-3, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {tke_temp.data(), dxt.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {X, 1, 1}, {X, Y, Z},					//shapes
		{2, 2, 0}, {2, 0, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {-2, 0, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);		

	
	inputs = {tke_temp.data(), maskW.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{2, 2, 0}, {2, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {-2, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {tke_temp.data(), cost.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {1, X, 1}, {X, Y, Z},					//shapes
		{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);		

	inputs = {tke_temp.data(), tke.data()}; //accumulate into tke, reuse tke_temp
	outputs = {tke.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		{2, 2, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},						//start index
		{-2, -2, 0}, {-2, -2, 0, -1}, {-2, -2, 0, -1},					//negativ end index
		devices, context, bins, q);	

	inputs = {flux_north.data(), flux_north.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{2, 2, 0}, {2, 1, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {-2, -3, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {tke_temp.data(), cost.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {tke_temp.data(), dyt.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("div3d", inputs, outputs, 
		{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {tke_temp.data(), maskW.data()};
	outputs = {tke_temp.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		{2, 2, 0}, {2, 2, 0,}, {2, 2, 0},						//start index
		{-2, -2, 0}, {-2, -2, 0,}, {-2, -2, 0},					//negativ end index
		devices, context, bins, q);

	inputs = {tke_temp.data(), tke.data()}; 
	outputs = {tke.data()};
	run_broadcast_kernel("add3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		{2, 2, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},						//start index
		{-2, -2, 0}, {-2, 2, 0, -1}, {-2, -2, 0, -1},					//negativ end index
		devices, context, bins, q);	

	// adv flux superbee wgrid unroll
	xt::xarray<double> maskUtr = xt::zeros_like(maskW);

	inputs = {maskW.data(), maskW.data()}; 
	outputs = {maskUtr.data()};
	run_broadcast_kernel("sub3d", inputs, outputs, 
		{X, Y, Z}, {X, Y, Z,}, {X, Y, Z,},					//shapes
		{0, 0, 0}, {1, 0, 0,}, {0, 0, 0,},						//start index
		{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0},					//negativ end index
		devices, context, bins, q);	

	inputs = {zero.data(), zero.data()}; //reeeeally need assignment kernel...
	outputs = {flux_east.data()};
	run_broadcast_kernel("mult3d", inputs, outputs, 
		{1}, {1}, {X, Y, Z,},					//shapes
		{0,}, {0,}, {0, 0, 0,},						//start index
		{0,}, {0,}, {0, 0, 0},					//negativ end index
		devices, context, bins, q);	

	adv_superbee(u, tke, maskUtr, dxt, 0, cost, cosu, devices, context, bins, q);

	std::cout << "sqrttke checksum: should be 1679...: " << xt::sum(sqrttke) << std::endl;
	std::cout << "ks checksum: should be 377...: " << xt::sum(ks) << std::endl;
	std::cout << "delta checksum: should be 85...: " << xt::sum(delta) << std::endl;
	std::cout << "diagonals checksum might change if mask is applied!!!" << std::endl;  
	std::cout << "a_tri checksum: should be 689.96 (392.9)...: " << xt::sum(a_tri) << std::endl;
	std::cout << "b_tri checksum: should be -629 (208.8)...: " << xt::sum(b_tri) << std::endl;
	std::cout << "b_tri_edge checksum: should be 527...: " << xt::sum(b_tri_edge) << std::endl;
	std::cout << "c_tri checksum: should be 835 (532.4)...: " << xt::sum(c_tri) << std::endl;
	std::cout << "d_tri checksum: should be 115.9 (-2157.7)...: " << xt::sum(d_tri) << std::endl;

	std::cout << "land_mask checksum: should be 584...:" << xt::sum(land_mask) << std::endl;
	std::cout << "edge_mask checksum: should be 584...:" << xt::sum(edge_mask) << std::endl;
	std::cout << "water_mask checksum: should be 1759...:" << xt::sum(water_mask) << std::endl;	
	
	std::cout << "not_edge_mask checksum: should be 2552...:" << xt::sum(not_edge_mask) << std::endl;
	std::cout << "gtsv doesnt work atm. we hijack checksums to follow wrong solution to implement rest of the code while debug...\n";
	std::cout << "tke checksum: should be -2052.5 (-926.5)...: " << xt::sum(tke) << std::endl;
	std::cout << "tke_tmp checksum: should be -2052.5 (-926.5)...: " << xt::sum(tke_temp) << std::endl;


	std::cout << "mask:.. (263) "<< xt::sum(mask) << std::endl;	
	std::cout << "tke surf corr (334.4).. " << xt::sum(tke_surf_corr) << std::endl;

	std::cout << "flux east (8060969).. " << xt::sum(flux_east) << std::endl;
	std::cout << "flux north (48331).. " << xt::sum(flux_north) << std::endl;



/*	for (int i=0; i<3136; i++){
		std::cout << a_tri[i] << ", ";
	}

	std::cout << "\n";

	for (int i=0; i<3136; i++){
		std::cout << b_tri[i] << ", ";
	}

	std::cout << "\n";

	for (int i=0; i<3136; i++){
		std::cout << c_tri[i] << ", ";
	}

	std::cout << "\n";

	for (int i=0; i<3136; i++){
		std::cout << d_tri[i] << ", ";
	}

	std::cout << "\n";

*/

	return 0;
}