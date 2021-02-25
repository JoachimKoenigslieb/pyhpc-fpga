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
#define SIZE = X * Y * Z


extern "C" {
    void dgtsv_(int*, int*, double*, double*, double*, double*, int*, int*);
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

double check_tridiag_solution(double* a_tri, double* b_tri, double* c_tri, double* d_tri, double* solution, int N, int verbose){
	double rhs_calculated;
	double error;

	rhs_calculated = b_tri[0] * solution[0] + c_tri[0] * solution[1];
	error += abs(rhs_calculated - d_tri[0]);
	if(verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[0] << " Accumulated error: " << error << " Solution found as pos " << 0 << ": " << solution[0] << std::endl;}

	for (int i=1; i<(N-1); i++){ //we neglect special cases
		rhs_calculated = a_tri[i] * solution[i-1] + b_tri[i] * solution[i] + c_tri[i] * solution[i+1];
		error += abs(rhs_calculated - d_tri[i]);
		if (verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[i] << " Accumulated error: " << error << " Solution found as pos " << i << ": " << solution[i] << std::endl;}
	}

	rhs_calculated = a_tri[N-1] * solution[N-2] + b_tri[N-1] * solution[N-1];
	error += abs( rhs_calculated - d_tri[N-1]);
	if (verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[N-1] << " Accumulated error: " << error << " Solution found as pos " << N-1 << ": " << solution[N-1] << std::endl;}

	return error;
}

int main(int argc, const char *argv[]){
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

	int N = 16; //should be 3136 or 4096. You can also compile a new kernel
	//N = next_largets_factor_2(N); // we try to pad to a factor of two in the hopes that this is what causes errors!

	xt::xarray<double> arg0 = xt::load_npy<double>("gtsv_arg0_size" + std::string(SIZE) + ".npy");
	xt::xarray<double> arg1 = xt::load_npy<double>("gtsv_arg1_size" + SIZE + ".npy");
	xt::xarray<double> arg2 = xt::load_npy<double>("gtsv_arg2_size" + SIZE + ".npy");
	xt::xarray<double> arg3 = xt::load_npy<double>("gtsv_arg3_size" + SIZE + ".npy");
	xt::xarray<double> arg3_copy = xt::zeros_like(arg3);

	for (int i=0; i<N; i++){
		arg3_copy[i] = arg3[i];
	}

	xt::xarray<double> res = xt::load_npy<double>("./gtsv_result_size " + SIZE + ".npy");

	std::vector<double*> inputs, outputs;

	inputs = {arg0.data(), arg1.data(), arg2.data(), arg3.data()};
	std::cout << xt::sum(arg0) << ", " << xt::sum(arg1) << ", " << xt::sum(arg2) << ", " << xt::sum(arg3) << std::endl;
	run_gtsv(N, inputs, devices, context, bins, q); //this outputs solution into d_tri 
	q.finish();

 	std::cout << "checksum scipy (lapack): \t\t" << xt::sum(res) << "\nvs computed fpga: \t" << xt::sum(arg3) << std::endl;

	std::cout << "check if this is a solution..: " << check_tridiag_solution(arg0.data(), arg1.data(), arg2.data(), arg3_copy.data(), arg3.data(), N, 0) << " should be low!";

	return 0;
}