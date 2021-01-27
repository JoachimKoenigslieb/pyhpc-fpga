#include <iostream>
#include <string.h>
#include "cnpy.h"
#include <iostream>

#include <random>
#include "xcl2.cpp"

extern "C" {
    void dgtsv_(int*, int*, double*, double*, double*, double*, int*, int*);
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

int main(){
	std::string xclbin_path = "./hw_emu_kernels.xclbin";
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
	devices.resize(1);

	int N = 10; //should be 3136 or 4096. You can also compile a new kernel
	//N = next_largets_factor_2(N); // we try to pad to a factor of two in the hopes that this is what causes errors!

	double* a_tri = aligned_alloc<double>(N);
	double* b_tri = aligned_alloc<double>(N);
	double* c_tri = aligned_alloc<double>(N);
	double* d_tri = aligned_alloc<double>(N);
	double* rhs_copy = aligned_alloc<double>(N);

	double err = 0.0;

	double lower_bound = 0;
	double upper_bound = 10;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re(4); //fixed seed

	int mode = 0;
	int verbose = 1;

	//random random data into tris and rhs. Save a copy of rhs, this will be overwritten in-place!
	if (mode == 0) {
		for (int i=0; i<N; i++){
			a_tri[i] = unif(re);
			b_tri[i] = unif(re);
			c_tri[i] = unif(re);
			d_tri[i] = unif(re);
			rhs_copy[i] = d_tri[i]; //save this, as d_tri will get written to in-place!
		}
	} else {
		for (int i = 0; i < N; i++) {
			a_tri[i] = -1.0;
			b_tri[i] = 2.0;
			c_tri[i] = -1.0;
			d_tri[i] = 0.0;
			rhs_copy[i] = d_tri[i];
		};

		d_tri[N - 1] = 1.0;
		d_tri[0] = 1.0;
		rhs_copy[0] = d_tri[0];
		rhs_copy[N -1] = d_tri[N -1];
	}

	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "a_tri", a_tri, {N}, "w");
	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "b_tri", b_tri, {N}, "a");
	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "c_tri", c_tri, {N}, "a");
	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "d_tri", d_tri, {N}, "a");
	std::cout << "Wrote diagonals...";

	std::vector<double*> inputs, outputs;

	inputs = {a_tri, b_tri, c_tri, d_tri};
	run_gtsv(N, inputs, devices, context, bins, q); //this outputs solution into d_tri 
	q.finish();
	std::cout << "The thing we get out of gtsv... : "; print_arr(d_tri, N); 

	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "xilinx_sol", d_tri, {N}, "a");
	
	double xilinx_error;

	xilinx_error = check_tridiag_solution(a_tri, b_tri, c_tri, rhs_copy, d_tri, N, verbose);

	double sum_solution = 0;
	for (int i=0; i<N; i++){
		sum_solution += d_tri[i];
	}
	std::cout << "accumulated error of solution /w xf::solver " << xilinx_error << std::endl;
 	std::cout << "sum of solution /w xf::solver: " << sum_solution << std::endl;

	//Now for lapack test:
	std::default_random_engine reMix(4); //fixed seed
	double* a_tri_copy = aligned_alloc<double>(N);
	double* b_tri_copy = aligned_alloc<double>(N);
	double* c_tri_copy = aligned_alloc<double>(N);

	if (mode == 0) {
		for (int i=0; i<N; i++){
			a_tri[i] = unif(reMix);
			b_tri[i] = unif(reMix);
			c_tri[i] = unif(reMix);
			d_tri[i] = unif(reMix);
			rhs_copy[i] = d_tri[i]; //save this, as d_tri will get written to in-place!
			a_tri_copy[i] = a_tri[i];
			b_tri_copy[i] = b_tri[i];
			c_tri_copy[i] = c_tri[i]; //lapack will destroy my data
		}
	} else {
		for (int i = 0; i < N; i++) {
			a_tri[i] = -1.0;
			b_tri[i] = 2.0;
			c_tri[i] = -1.0;
			d_tri[i] = 0.0;
			rhs_copy[i] = d_tri[i];
			a_tri_copy[i] = a_tri[i];
			b_tri_copy[i] = b_tri[i];
			c_tri_copy[i] = c_tri[i]; //lapack will destroy my data
		};

		d_tri[N - 1] = 1.0;
		d_tri[0] = 1.0;
		rhs_copy[0] = d_tri[0];
		rhs_copy[N -1] = d_tri[N -1];
	}

	a_tri[0] = 0.0;
	c_tri[N-1] = 0.0;

	int info = 0;
	int NHRS = 1;
	int LDB = N; 

	dgtsv_(&N, &NHRS, &a_tri[1], b_tri, c_tri, d_tri, &LDB, &info); //call lapack
	cnpy::npz_save("tridiag_data"+std::to_string(N) +".npz", "lapack_sol", d_tri, {N}, "a");

	double lapack_error;

	lapack_error = check_tridiag_solution(a_tri_copy, b_tri_copy, c_tri_copy, rhs_copy, d_tri, N, verbose);

	sum_solution = 0;
	for (int i=0; i<N; i++){
		sum_solution += d_tri[i];
	}

	std::cout << "accumulated error of solution /w lapack " << lapack_error << std::endl;
	std::cout << "sum of solution /w lapack: " << sum_solution << std::endl;
 
	return 0;
}
