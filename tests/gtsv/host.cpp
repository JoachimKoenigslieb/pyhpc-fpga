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

int next_largets_factor_2(int n){
	int factor_2 = 1;  
	while (factor_2 < n){
		factor_2 *= 2;
	}
	return factor_2;
}

void print_arr(double* start, int n){
	for (int i=0; i<n; i++){
		std::cout << start[i] << ", ";
	}
	std::cout << std::endl;
}

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

void run_gtsv(int kernel_size, std::vector<double *> &inputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
	// tridiagnals are taken to be equal length. We intepret upper and lower diagonals such that the extra index is a zero by writng that memory to zero. 
	int N_pow_2 = next_largets_factor_2(kernel_size); // For some reason, gtsv kernel only works when its powers of two sized input. We are going to zero pad!
	std::string kernel_name = "gtsv" + std::to_string(N_pow_2);

	std::cout << "Running kernel: " << kernel_name << std::endl;

	cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
	cl::Kernel kernel(program, kernel_name.data());

	// DDR Settings
	std::vector<cl_mem_ext_ptr_t> mext_io(4);
	mext_io[0].flags = XCL_MEM_DDR_BANK3;
	mext_io[1].flags = XCL_MEM_DDR_BANK3;
	mext_io[2].flags = XCL_MEM_DDR_BANK3;
	mext_io[3].flags = XCL_MEM_DDR_BANK3;


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

	a_tri_padded[0] = 0;
	c_tri_padded[kernel_size - 1] = 0;

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
	std::cout << "INFO: Finish data transfer from host to device" << std::endl;

	// Setup kernel
	kernel.setArg(0, N_pow_2);
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

	for (int i=0; i<kernel_size; i++){ // put the solution from the padded output of kernel into working memeory
		d_tri[i] = d_tri_padded[i];
	}
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

	xt::xarray<double> arg0 = xt::load_npy<double>("gtsv_arg0.npy");
	xt::xarray<double> arg1 = xt::load_npy<double>("gtsv_arg1.npy");
	xt::xarray<double> arg2 = xt::load_npy<double>("gtsv_arg2.npy");
	xt::xarray<double> arg3 = xt::load_npy<double>("gtsv_arg3.npy");
	xt::xarray<double> arg3_copy = xt::zeros_like(arg3);

	for (int i=0; i<N; i++){
		arg3_copy[i] = arg3[i];
	}

	xt::xarray<double> res = xt::load_npy<double>("./gtsv_result.npy");

	std::vector<double*> inputs, outputs;

	inputs = {arg0.data(), arg1.data(), arg2.data(), arg3.data()};
	std::cout << xt::sum(arg0) << ", " << xt::sum(arg1) << ", " << xt::sum(arg2) << ", " << xt::sum(arg3) << std::endl;
	run_gtsv(N, inputs, devices, context, bins, q); //this outputs solution into d_tri 
	q.finish();

 	std::cout << "checksum scipy (lapack): \t\t" << xt::sum(res) << "\nvs computed fpga: \t" << xt::sum(arg3) << std::endl;

	std::cout << "check if this is a solution..: " << check_tridiag_solution(arg0.data(), arg1.data(), arg2.data(), arg3_copy.data(), arg3.data(), N, 0) << " should be low!";

	return 0;
}
