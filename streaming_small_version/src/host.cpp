#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

#include <string.h>
#include <xcl2/xcl2.cpp>
#include <xsimd/xsimd.hpp>
#include <xcl2/xcl2.hpp> 

#include <runKernelsEdge.cpp>
#include "CL/cl_ext_xilinx.h"

using namespace xt::placeholders; //enables xt::range(1, _) syntax. eqiv. to [1:] syntax in numpy 
namespace xs = xsimd;

void print_sum(double* arr, int N){
	double s=0;
	for (int i=0; i<N; i++){
		std::cout << arr[i] << ", ";
		s += arr[i];
	}
	std::cout << "sum: " << s << std::endl;
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
	ArgParser parser(argc, argv);
	
	std::string xclbin_path;
	std::string size;
	std::string X_str, Y_str, Z_str;
	
	if (!parser.getCmdOption("-xclbin", xclbin_path)){
		std::cout << "please set -xclbin path!" << std::endl;
	}

	if (!parser.getCmdOption("-size", size)){
		std::cout << "please set -size paramter to a matching .npy file" << std::endl;
	}

	if (!parser.getCmdOption("-X", X_str)){
		std::cout << "please set -X paramter to a matching .npy file" << std::endl;
	}

	if (!parser.getCmdOption("-Y", Y_str)){
		std::cout << "please set -Y paramter to a matching .npy file" << std::endl;
	}

	if (!parser.getCmdOption("-Z", Z_str)){
		std::cout << "please set -Z paramter to a matching .npy file" << std::endl;
	}

	int X, Y, Z;
	X = std::stoi(X_str);
	Y = std::stoi(Y_str);
	Z = std::stoi(Z_str);

	std::cout << "running " << xclbin_path << " for inputs sized " << size << std::endl;
	
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!

	devices.resize(1);

	std::vector<double *> inputs, outputs;

	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> u = xt::load_npy<double>("./np_files/u.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> v = xt::load_npy<double>("./np_files/v.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> w = xt::load_npy<double>("./np_files/w.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> maskU = xt::load_npy<double>("./np_files/maskU.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> maskV = xt::load_npy<double>("./np_files/maskV.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> maskW = xt::load_npy<double>("./np_files/maskW.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dxt = xt::load_npy<double>("./np_files/dxt.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dxu = xt::load_npy<double>("./np_files/dxu.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dyt = xt::load_npy<double>("./np_files/dyt.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dyu = xt::load_npy<double>("./np_files/dyu.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dzt = xt::load_npy<double>("./np_files/dzt.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dzw = xt::load_npy<double>("./np_files/dzw.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> cost = xt::load_npy<double>("./np_files/cost.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> cosu = xt::load_npy<double>("./np_files/cosu.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> kbot = xt::load_npy<double>("./np_files/kbot.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> forc_tke_surface = xt::load_npy<double>("./np_files/forc_tke_surface.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> kappaM = xt::load_npy<double>("./np_files/kappaM.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> mxl = xt::load_npy<double>("./np_files/mxl.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> forc = xt::load_npy<double>("./np_files/forc.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> tke = xt::load_npy<double>("./np_files/tke.npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> dtke = xt::load_npy<double>("./np_files/dtke.npy");

	int SIZE = X * Y * Z;

	double* arrays_1d = aligned_alloc<double>(8 * X);
	double* arrays_2d = aligned_alloc<double>(2 * X * Y);
	double* arrays_3d = aligned_alloc<double>(6 * SIZE);
	double* arrays_4d = aligned_alloc<double>(5 * SIZE * 3);
	double* output = aligned_alloc<double>(SIZE * 3);

	memcpy(arrays_1d, dxt.data(), sizeof(double) * X);
	memcpy(arrays_1d + X, dxu.data(), sizeof(double) * X);
	memcpy(arrays_1d + 2*X, dyt.data(), sizeof(double) * X);
	memcpy(arrays_1d + 3*X, dyu.data(), sizeof(double) * X);
	memcpy(arrays_1d + 4*X, dzt.data(), sizeof(double) * X);
	memcpy(arrays_1d + 5*X, dzw.data(), sizeof(double) * X);
	memcpy(arrays_1d + 6*X, cosu.data(), sizeof(double) * X);
	memcpy(arrays_1d + 7*X, cost.data(), sizeof(double) * X);
  
	memcpy(arrays_2d, kbot.data(), sizeof(double) * X * Y);
	memcpy(arrays_2d + (X * Y), forc_tke_surface.data(), sizeof(double) * X * Y);

	memcpy(arrays_3d, kappaM.data(), sizeof(double) * SIZE);
	memcpy(arrays_3d + SIZE, mxl.data(), sizeof(double) * SIZE);
	memcpy(arrays_3d + (SIZE * 2), forc.data(), sizeof(double) * SIZE);
	memcpy(arrays_3d + (SIZE * 3), maskU.data(), sizeof(double) * SIZE);
	memcpy(arrays_3d + (SIZE * 4), maskV.data(), sizeof(double) * SIZE);
	memcpy(arrays_3d + (SIZE * 5), maskW.data(), sizeof(double) * SIZE);

	memcpy(arrays_4d, u.data(), sizeof(double) * SIZE * 3);
	memcpy(arrays_4d + SIZE * 3, v.data(), sizeof(double) * SIZE * 3);
	memcpy(arrays_4d + (SIZE * 3 * 2), w.data(), sizeof(double) * SIZE * 3);
	memcpy(arrays_4d + (SIZE * 3 * 3), tke.data(), sizeof(double) * SIZE * 3);
	memcpy(arrays_4d + (SIZE * 3 * 4), dtke.data(), sizeof(double) * SIZE * 3);

	inputs = { arrays_1d, arrays_2d, arrays_3d, arrays_4d, };  
	outputs = {output};
	std::cout << "enqueue kernel...\n";
	run_broadcast_kernel("work", inputs, outputs, {X, Y, Z}, devices, context, bins, q, program);
	std::cout << "recieved data on host\n";
	print_sum(output, 6*6*6*3);
	return 0;
}
