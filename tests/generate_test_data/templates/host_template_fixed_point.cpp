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

#include <runKernels.h>

#define X <X>
#define Y <Y>
#define Z <Z>
#define SIZE X * Y * Z

// this is only in fixed points hopefully!!!
using namespace xt::placeholders; //enables xt::range(1, _) syntax. eqiv. to [1:] syntax in numpy 

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

	// Init of FPGA device
	
	if (!parser.getCmdOption("-xclbin", xclbin_path)){
		std::cout << "please set -xclbin path!" << std::endl;
	}
	
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
	devices.resize(1);

	<loadArgs>
	xt::xarray<double> res = xt::load_npy<double>("./<func>_result_size" + std::to_string(SIZE) + ".npy");
	xt::xarray<double> res_compute = xt::zeros_like(res);

	std::vector<double *> inputs, outputs;

	inputs = {<args>}; 
	outputs = {res_compute.data()};
	<kernelRun>

	std::cout << "checksum numpy: \t\t" << xt::sum(res) << "\nvs computed fpga: \t" << xt::sum(res_compute) << std::endl;

	return 0;
}
