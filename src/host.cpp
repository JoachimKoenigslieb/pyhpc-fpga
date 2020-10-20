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

// Compute time difference
unsigned long diff(const struct timeval *newTime, const struct timeval *oldTime)
{
	return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}

void run_kernel(std::string kernel_name, int kernel_size, std::vector<double *> &inputs, std::vector<double *> &outputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
	// this is a helper function to execute a kernel.
	{
		cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
		cl::Kernel kernel(program, kernel_name.data());

		std::cout << "INFO: Kernel '" << kernel_name << "' has been created" << std::endl;

		int num_in = inputs.size();
		int num_out = outputs.size();

		std::vector<cl::Buffer> in_buffers(num_in);
		std::vector<cl::Buffer> out_buffers(num_out);

		for (int i = 0; i < num_in; i++)
		{
			in_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(double) * kernel_size, inputs[i]);
		}

		for (int i = 0; i < num_out; i++)
		{
			out_buffers[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(double) * kernel_size, outputs[i]);
		}

		std::cout << "INFO: Buffers has been created" << std::endl;

		kernel.setArg(0, kernel_size); //we need to set arg0 to size!

		for (int i = 0; i < num_in; i++)
		{
			kernel.setArg(i + 1, in_buffers[i]);
			q.enqueueMigrateMemObjects({in_buffers[i]}, 0);
		}

		for (int i = 0; i < num_out; i++)
		{
			kernel.setArg(i + num_in + 1, out_buffers[i]);
		}

		std::cout << "INFO: Arguments has been set" << std::endl;

		std::cout << "INFO: Migrated to device" << std::endl;

		q.enqueueTask(kernel);

		q.finish();

		q.enqueueMigrateMemObjects({out_buffers[0]}, CL_MIGRATE_MEM_OBJECT_HOST); // 1 : migrate from dev to host

		std::cout << "INFO: Migrated to back to host" << std::endl;

		q.finish();
	}
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
	std::string xclbin_path = "./kernels.xclbin";
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

	// Initialization of data

	/* i think this is not needed. idk why i dont get mem errors for using non aligned memory 
	double *u_data = aligned_alloc<double>(size_4d);
	double *v_data = aligned_alloc<double>(size_4d);
	double *w_data = aligned_alloc<double>(size_4d);
	double *maskU_data = aligned_alloc<double>(size_3d);
	double *maskV_data = aligned_alloc<double>(size_3d);
	double *maskW_data = aligned_alloc<double>(size_3d);
	double *dxt_data = aligned_alloc<double>(size_1d_hoz);
	double *dxu_data = aligned_alloc<double>(size_1d_hoz);
	double *dyt_data = aligned_alloc<double>(size_1d_hoz);
	double *dyu_data = aligned_alloc<double>(size_1d_hoz);
	double *dzt_data = aligned_alloc<double>(size_1d_vert);
	double *dzw_data = aligned_alloc<double>(size_1d_vert);
	double *cost_data = aligned_alloc<double>(size_1d_hoz);
	double *cosu_data = aligned_alloc<double>(size_1d_hoz);
	double *kbot_data = aligned_alloc<double>(size_2d);
	double *forc_tke_surface_data = aligned_alloc<double>(size_2d);
	double *kappaM_data = aligned_alloc<double>(size_3d);
	double *mxl_data = aligned_alloc<double>(size_3d);
	double *forc_data = aligned_alloc<double>(size_3d);
	double *tke_data = aligned_alloc<double>(size_4d);
	double *dtke_data = aligned_alloc<double>(size_4d);
	double *flux_east_data = aligned_alloc<double>(size_3d);
	double *flux_north_data = aligned_alloc<double>(size_3d);
	double *flux_top_data = aligned_alloc<double>(size_3d);
	double *temp_data_3d = aligned_alloc<double>(size_3d);
	double *sqrttke_data = aligned_alloc<double>(size_3d);
	double *a_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *b_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *c_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *d_tri_data = aligned_alloc<double>((Y-2) * (X - 2));
	double *delta_data = aligned_alloc<double>((Y-2) * (X - 2));
	*/

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
	xt::xarray<double> a_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> b_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> c_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> d_tri = xt::zeros<double>({X-2, Y-2, Z});
	xt::xarray<double> delta = xt::zeros<double>({X-2, Y-2, Z});


	/* this block might be not needed!
	u_data = u.data();
	v_data = v.data();
	w_data = w.data();
	maskU_data = maskU.data();
	maskV_data = maskV.data();
	maskW_data = maskW.data();
	dxt_data = dxt.data();
	dxu_data = dxu.data();
	dyt_data = dyt.data();
	dyu_data = dyu.data();
	dzt_data = dzt.data();
	dzw_data = dzw.data();
	cost_data = cost.data();
	cosu_data = cosu.data();
	kbot_data = kbot.data();
	forc_tke_surface_data = forc_tke_surface.data();
	kappaM_data = kappaM.data();
	mxl_data = mxl.data();
	forc_data = forc.data();
	tke_data = tke.data();
	dtke_data = dtke.data();
	flux_east_data = flux_east.data();
	flux_north_data = flux_north.data();
	flux_top_data = flux_top.data();
	sqrttke_data = sqrttke.data();
	a_tri_data = a_tri.data();
	b_tri_data = b_tri.data();
	c_tri_data = c_tri.data();
	d_tri_data = d_tri.data();
	delta_data = delta.data();
	*/


	int tau = 0;
	double taup1 = 1.;
	double taum1 = 2.;
	double dt_tracer = 1.;
	double dt_mom = 1.;
	double AB_eps = 0.1;
	double alpha_tke = 1.;
	double c_eps = 0.7;
	double K_h_tke = 2000.;

	std::vector<double *> inputs;
	std::vector<double *> outputs;


	inputs = {
				xt::view(tke, xt::all(), xt::all(), xt::all(), tau).data(), 
				flux_east.data()
			}; 

	outputs = {sqrttke.data()};
	run_kernel("vmax", size_3d, inputs, outputs, devices, context, bins, q);

	inputs = {sqrttke.data()};
	outputs = {sqrttke.data()};
	run_kernel("vsqrt", size_3d, inputs, outputs, devices, context, bins, q);

	xt::xarray<double> ones = xt::ones<double>({X-2, Y-2, Z});

	xt::xarray<double> cs = xt::sum(delta);

	inputs = {	
				ones.data(),
				xt::view(dzt, xt::newaxis(), xt::newaxis(), xt::range(1, _)).data()
			};
	outputs = { xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data() };
	run_kernel("vdiv", (X-2)*(Y-2)*Z, inputs, outputs, devices, context, bins, q);

	std::cout << dzt << std::endl;

	inputs = { 
				xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data(), 
				xt::eval(ones * 0.5 * alpha_tke * dt_mom).data() // we need eval as xtensor is lazy!
			};
	outputs = { xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data() };
	run_kernel("vmult", (X-2)*(Y-2)*Z, inputs, outputs, devices, context, bins, q); //this is the most useless op. we multiple by a scalar, but we allready do the multiplication on the host anyways.. (ew)

	inputs = {
				xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data(), 
				xt::view(kappaM, xt::range(2, -2), xt::range(2, -2), xt::range(_,-1)).data()
			};
	outputs = { xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data() };
	run_kernel("vmult", (X-2)*(Y-2)*Z, inputs, outputs, devices, context, bins, q);

	inputs = {
				xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data(), 
				xt::view(kappaM, xt::range(2, -2), xt::range(2, -2), xt::range(1,_)).data()
			};
	outputs = { xt::view(delta, xt::all(), xt::all(), xt::range(_, -1)).data() };
	run_kernel("vadd", (X-2)*(Y-2)*Z, inputs, outputs, devices, context, bins, q);

	cs = xt::sum(delta);

	std::cout << cs << std::endl;

	return 0;
}
