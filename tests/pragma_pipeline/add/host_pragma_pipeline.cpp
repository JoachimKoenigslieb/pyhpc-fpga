#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

#include <xcl2/xcl2.cpp>
#include <xsimd/xsimd.hpp>

#include <runKernels.h>

using namespace xt::placeholders; //enables xt::range(1, _) syntax. eqiv. to [1:] syntax in numpy 
namespace xs = xsimd;

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
	devices.resize(1);

	xt::xarray<double> res = xt::load_npy<double>("./npfiles/add_result_size" + size + ".npy");

	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> arg0 = xt::load_npy<double>("add_arg0_size" + size + ".npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> arg1 = xt::load_npy<double>("add_arg1_size" + size + ".npy");
	xt::xarray<double, XTENSOR_DEFAULT_LAYOUT, xsimd::aligned_allocator<double, 4096>> res_compute = xt::zeros_like(res);

	std::vector<double *> inputs, outputs;

	inputs = {arg0.data(), arg1.data()}; 
	outputs = {res_compute.data()};
	run_broadcast_kernel("add4d", inputs, outputs, 
		{X, Y, Z,}, 		{X, Y, Z,}, 		{X, Y, Z,},
		{0, 0, 0,}, 		{0, 0, 0,}, 		{0, 0, 0,},
		{0, 0, 0,}, 		{0, 0, 0,}, 		{0, 0, 0,},
devices, context, bins, q);

	std::cout << "checksum numpy: \t\t" << xt::sum(res) << "\nvs computed fpga: \t" << xt::sum(res_compute) << std::endl;

	return 0;
}
