#include <iostream>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <xcl2/xcl2.cpp>
#include <runKernels.h>

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


void create_data(double* A, double* B, double* result, int N){
	for (int i=0; i<N; i++){
		A[i] = 1.0;
		B[i] = 2.0;
		result[i] = 3.0;
	}
}

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

	double* arg0 = new double [X*Y*Z]; 
	double* arg1 = new double [X*Y*Z];
	double* res = new double [X*Y*Z];
	double* res_compute = new double [X*Y*Z];

	create_data(arg0, arg1, res, X*Y*Z);

	std::vector<double *> inputs, outputs;

	inputs = {arg0, arg1}; 
	outputs = {res_compute};
	run_broadcast_kernel("add4d", inputs, outputs, 
		{X, Y, Z,}, 		{X, Y, Z,}, 		{X, Y, Z,},
		{0, 0, 0,}, 		{0, 0, 0,}, 		{0, 0, 0,},
		{0, 0, 0,}, 		{0, 0, 0,}, 		{0, 0, 0,},
devices, context, bins, q);

	return 0;
}
