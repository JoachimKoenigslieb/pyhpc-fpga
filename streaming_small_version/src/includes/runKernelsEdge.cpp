#include <vector>
#include <iostream>
#include <string>
#include <assert.h>
#include <xcl2/xcl2.hpp>

void run_broadcast_kernel(std::string kernel_name,
							std::vector<double *> &inputs,
							std::vector<double *> &outputs,
							std::vector<int> XYZ,
							std::vector<cl::Device> &devices,
							cl::Context &context,
							cl::Program::Binaries &bins,
							cl::CommandQueue &q,
							cl::Program &program)
{
	cl::Kernel kernel(program, kernel_name.data());

	int size_1d = XYZ[0];
	int size_2d = XYZ[0] * XYZ[1];
	int size_3d = XYZ[0] * XYZ[1] * XYZ[2];
	int size_4d = XYZ[0] * XYZ[1] * XYZ[2] * 3; 

	cl::Buffer arrays_buffer_1d = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * size_1d * 8, inputs[0]);
	cl::Buffer arrays_buffer_2d = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * size_2d * 2, inputs[1]);
	cl::Buffer arrays_buffer_3d = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * size_3d * 6, inputs[2]);
	cl::Buffer arrays_buffer_4d = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * size_4d * 5, inputs[3]);
	cl::Buffer output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double) * size_4d, outputs[0]);

	kernel.setArg(0, arrays_buffer_1d);
	kernel.setArg(1, arrays_buffer_2d);
	kernel.setArg(2, arrays_buffer_3d);
	kernel.setArg(3, arrays_buffer_4d);
	kernel.setArg(4, output_buffer);

	q.enqueueMigrateMemObjects({arrays_buffer_1d, arrays_buffer_2d, arrays_buffer_3d, arrays_buffer_4d}, 0);
	q.enqueueTask(kernel);

	q.finish();
	q.enqueueMigrateMemObjects({output_buffer}, CL_MIGRATE_MEM_OBJECT_HOST);
	q.finish();
}
