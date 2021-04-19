#include "runKernelsFixed.h"
#include <vector>
#include <iostream>
#include <string>
#include <assert.h>
#include <xcl2/xcl2.hpp>
#include "ap_fixed.h"

typedef ap_ufixed<19, 11> data_in;
typedef ap_ufixed<20, 12> data_out;

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

void pad_end(std::vector<int> &vec, int val, int to_len){
	int vec_len = vec.size();
	for (int i=0; i<(to_len - vec_len); i++){
		vec.push_back(val);
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
	std::vector<int> new_stride;

	new_stride = zero_on_squeeze(view_shape, stride);
	pad_begin(new_stride, 0, out_dim);

	return new_stride;
}

std::vector<int> rebuild_offset(std::vector<int> offset, std::vector<int> view_shape, std::vector<int> out_offset, int out_dim){
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

	pad_end(A_stride_res, 0, 4);
	pad_end(A_shape, 1, 4);
	pad_end(A_offset, 0, 4);
	pad_end(A_end_offset, 0, 4);

	pad_end(out_stride_res, 0, 4);
	pad_end(out_shape, 1, 4);
	pad_end(out_offset, 0, 4);
	pad_end(out_end_offset, 0, 4);

	out_dim = 4;
}

int next_largets_factor_2(int n){
	int factor_2 = 1;  
	while (factor_2 < n){
		factor_2 *= 2;
	}
	return factor_2;
}

void run_1d_kernel(std::string kernel_name,
							std::vector<double *> &inputs,
							std::vector<double *> &outputs,
							std::vector<int> A_shape,
							std::vector<int> output_shape,
							std::vector<int> A_offset,
							std::vector<int> output_offset,
							std::vector<int> A_offset_end,
							std::vector<int> output_offset_end,
							std::vector<cl::Device> &devices,
							cl::Context &context,
							cl::Program::Binaries &bins,
							cl::CommandQueue &q)
{
	//setup program and kernel:
	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
	cl::Kernel kernel(program, kernel_name.data());

	int num_in = 1;
	int dimensions;

	cl::Buffer output_buffer;
	cl::Buffer A_buffer;
	cl::Buffer strides_offsets_buffer;

	std::vector<int> A_stride, output_stride;
	int A_lin_offset, output_lin_offset;
	int output_data_size, A_data_size;

	negotiate_strides(A_shape, output_shape, A_offset, output_offset, A_offset_end, output_offset_end, A_stride, output_stride, A_lin_offset, output_lin_offset, dimensions, A_data_size, output_data_size);

	// [strides] [offsets] [out shape] [out end offset]
	// 3*D      + 4d      + D         + D
	// Total length is 8D where D is dimension of output
	int* strides_offsets = aligned_alloc<int>(8 * dimensions); // allocate alligned for 8 integers.


	for (int i = 0; i<dimensions; i++){
		strides_offsets[i] = A_stride[i];
		strides_offsets[i+2*dimensions] = output_stride[i];

		strides_offsets[i+3*dimensions] = A_offset[i];
		strides_offsets[i+5*dimensions] = output_offset[i];

		strides_offsets[i+6*dimensions] = output_shape[i];
		strides_offsets[i+7*dimensions] = output_offset_end[i];
	}

	A_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * A_data_size, inputs[0]);

	output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * output_data_size, outputs[0]);
	strides_offsets_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * 8 * dimensions, strides_offsets);

	assert(output_shape.size() == dimensions);
	assert(output_stride.size() == dimensions);
	assert(output_offset.size() == dimensions);
	assert(output_offset_end.size() == dimensions);

	kernel.setArg(0, A_buffer);
	kernel.setArg(1, output_buffer);
	kernel.setArg(2, A_lin_offset);
	kernel.setArg(3, output_lin_offset);
	kernel.setArg(4, strides_offsets_buffer);
	kernel.setArg(5, dimensions);

	q.enqueueMigrateMemObjects({A_buffer}, 0);
	q.enqueueMigrateMemObjects({output_buffer}, 0);
	q.enqueueMigrateMemObjects({strides_offsets_buffer}, 0);

	q.finish();

	q.enqueueTask(kernel);
	q.finish();
	q.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

	q.finish();
}

void run_broadcast_kernel(std::string kernel_name,
							std::vector<data_in *> &inputs,
							std::vector<data_out *> &outputs,
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
	//setup program and kernel:
	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
	cl::Kernel kernel(program, kernel_name.data());

	std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

	int num_in = 2;
	int dimensions;

	cl::Buffer output_buffer;
	cl::Buffer A_buffer;
	cl::Buffer B_buffer;
	cl::Buffer strides_offsets_buffer;

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

	// [strides] [offsets] [out shape] [out end offset]
	// 3*D      + 4d      + D         + D
	// Total length is 8D where D is dimension of output
	int* strides_offsets = aligned_alloc<int>(8 * dimensions); // allocate alligned for 8 integers.

	for (int i = 0; i<dimensions; i++){
		strides_offsets[i] = A_stride[i];
		strides_offsets[i+1*dimensions] = B_stride[i];
		strides_offsets[i+2*dimensions] = output_stride[i];

		strides_offsets[i+3*dimensions] = A_offset[i];
		strides_offsets[i+4*dimensions] = B_offset[i];
		strides_offsets[i+5*dimensions] = output_offset[i];

		strides_offsets[i+6*dimensions] = output_shape[i];
		strides_offsets[i+7*dimensions] = output_offset_end[i];
	}

	A_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(data_in) * A_data_size, inputs[0]);
	B_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(data_out) * B_data_size, inputs[1]);

	output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * output_data_size, outputs[0]);
	strides_offsets_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * 8 * dimensions, strides_offsets);

	assert(output_shape.size() == dimensions);
	assert(output_stride.size() == dimensions);
	assert(output_offset.size() == dimensions);
	assert(output_offset_end.size() == dimensions);

	kernel.setArg(0, A_buffer);
	kernel.setArg(1, B_buffer);
	kernel.setArg(2, output_buffer);
	kernel.setArg(3, A_lin_offset);
	kernel.setArg(4, B_lin_offset);
	kernel.setArg(5, output_lin_offset);
	kernel.setArg(6, strides_offsets_buffer);
	kernel.setArg(7, dimensions);

	q.enqueueMigrateMemObjects({A_buffer}, 0);
	q.enqueueMigrateMemObjects({B_buffer}, 0);
	q.enqueueMigrateMemObjects({output_buffer}, 0);
	q.enqueueMigrateMemObjects({strides_offsets_buffer}, 0);

	q.finish();

	q.enqueueTask(kernel);
	q.finish();
	q.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

	q.finish();
}

void run_gtsv(int kernel_size, std::vector<double *> &inputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
	// tridiagnals are taken to be equal length. We intepret upper and lower diagonals such that the extra index is a zero by writng that memory to zero. 
	int N_pow_2 = next_largets_factor_2(kernel_size); // For some reason, gtsv kernel only works when its powers of two sized input. We are going to zero pad!
	std::string kernel_name = "gtsv" + std::to_string(N_pow_2);

	std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
	cl::Kernel kernel(program, kernel_name.data());

	// DDR Settings
	std::vector<cl_mem_ext_ptr_t> mext_io(4);
	mext_io[0].flags = XCL_MEM_DDR_BANK0;
	mext_io[1].flags = XCL_MEM_DDR_BANK0;
	mext_io[2].flags = XCL_MEM_DDR_BANK0;
	mext_io[3].flags = XCL_MEM_DDR_BANK0;

	inputs[0][0] = 0; 	//on a tri-diagonal matrix, upper and lower diagonals have n-1 entries. We zero them like this to fit gtsv kernel convetion
	inputs[2][kernel_size - 1] = 0;

	double* a_tri_padded = aligned_alloc<double>(N_pow_2);
	double* b_tri_padded = aligned_alloc<double>(N_pow_2);
	double* c_tri_padded = aligned_alloc<double>(N_pow_2);
	double* d_tri_padded = aligned_alloc<double>(N_pow_2);

	double *a_tri = inputs[0];
	double *b_tri = inputs[1];
	double *c_tri = inputs[2];
	double *d_tri = inputs[3];

	for (int i = 0; i<kernel_size; i++){
		a_tri_padded[i] = a_tri[i];
		b_tri_padded[i] = b_tri[i];
		c_tri_padded[i] = c_tri[i];
		d_tri_padded[i] = d_tri[i];

	}
	for (int i = kernel_size; i<N_pow_2; i++){
		a_tri_padded[i] = 0;
		b_tri_padded[i] = 1; // i think this helps with stability,
		c_tri_padded[i] = 0;
		d_tri_padded[i] = 0;
	}

	mext_io[0].obj = a_tri_padded;
	mext_io[0].param = 0;
	mext_io[1].obj = b_tri_padded;
	mext_io[1].param = 0;
	mext_io[2].obj = c_tri_padded;
	mext_io[2].param = 0;
	mext_io[3].obj = d_tri_padded;
	mext_io[3].param = 0;

	// Create device buffer and map dev buf to host buf
	cl::Buffer matdiaglow_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
									sizeof(double) * N_pow_2, &mext_io[0]);
	cl::Buffer matdiag_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
								sizeof(double) * N_pow_2, &mext_io[1]);
	cl::Buffer matdiagup_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
									sizeof(double) * N_pow_2, &mext_io[2]);
	cl::Buffer rhs_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
							sizeof(double) * N_pow_2, &mext_io[3]);

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

	// Setup kernel
	kernel.setArg(0, N_pow_2);
	kernel.setArg(1, matdiaglow_buffer);
	kernel.setArg(2, matdiag_buffer);
	kernel.setArg(3, matdiagup_buffer);
	kernel.setArg(4, rhs_buffer);
	q.finish();

	q.enqueueTask(kernel, nullptr, nullptr);
	q.finish();
	q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr); // 1 : migrate from dev to host
	q.finish();

	for (int i=0; i<kernel_size; i++){ // put the solution from the padded output of kernel into working memeory
		d_tri[i] = d_tri_padded[i];
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
	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
	cl::Kernel kernel(program, kernel_name.data());
	std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

	//num inputs outputs
	int num_in = inputs.size();
	int dimensions;
	int num_out = outputs.size();

	//setup buffers:
	cl::Buffer output_buffer;
	cl::Buffer A_buffer;
	cl::Buffer B_buffer;
	cl::Buffer C_buffer;
	cl::Buffer strides_offsets_buffer;

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

	// [strides] [offsets] [out shape] [out end offset]
	// 4*D      + 4*D      + D         + D
	// Total length is 10D where D is dimension of output
	// std::vector<int> strides_offsets(10 * dimensions);
	int* strides_offsets = aligned_alloc<int>(10 * dimensions); // allocate alligned for 8 integers.



	for (int i = 0; i<dimensions; i++){
		strides_offsets[i] = A_stride[i];
		strides_offsets[i+1*dimensions] = B_stride[i];
		strides_offsets[i+2*dimensions] = C_stride[i];
		strides_offsets[i+3*dimensions] = output_stride[i];

		strides_offsets[i+4*dimensions] = A_offset[i];
		strides_offsets[i+5*dimensions] = B_offset[i];
		strides_offsets[i+6*dimensions] = C_offset[i];
		strides_offsets[i+7*dimensions] = output_offset[i];

		strides_offsets[i+8*dimensions] = output_shape[i];
		strides_offsets[i+9*dimensions] = output_offset_end[i];
	}

	A_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * A_data_size, inputs[0]);
	B_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * B_data_size, inputs[1]);
	C_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * C_data_size, inputs[2]);

	output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * output_data_size, outputs[0]);
	strides_offsets_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int) * 10 * dimensions, strides_offsets);

	assert(output_shape.size() == dimensions);
	assert(output_stride.size() == dimensions);
	assert(output_offset.size() == dimensions);
	assert(output_offset_end.size() == dimensions);

	kernel.setArg(0, A_buffer);
	kernel.setArg(1, B_buffer);
	kernel.setArg(2, C_buffer);
	kernel.setArg(3, output_buffer);
	kernel.setArg(4, A_lin_offset);
	kernel.setArg(5, B_lin_offset);
	kernel.setArg(6, C_lin_offset);
	kernel.setArg(7, output_lin_offset);
	kernel.setArg(8, strides_offsets_buffer);
	kernel.setArg(9, dimensions);


	q.enqueueMigrateMemObjects({A_buffer}, 0);
	q.enqueueMigrateMemObjects({B_buffer}, 0);
	q.enqueueMigrateMemObjects({C_buffer}, 0);
	q.enqueueMigrateMemObjects({output_buffer}, 0);
	q.enqueueMigrateMemObjects({strides_offsets_buffer}, 0);

	q.finish();

	q.enqueueTask(kernel);
	q.finish();
	q.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

	q.finish();
}
