#include <cmath>
#include <iostream>
#include <string.h>
#include <xf_solver/xf_solver_L2.hpp>

#define NRC 32 // should be a power of 2, or program will result in not correct values
#define NCU 1


extern "C" void gtsv(int n, double* matDiagLow, double* matDiag, double* matDiagUp, double* rhs, int debug=0) {
	double a_padded[NRC];
	double b_padded[NRC];
	double c_padded[NRC];
	double d_padded[NRC];

	memcpy(a_padded, matDiagLow, n * sizeof(double));
	memcpy(b_padded, matDiag, n * sizeof(double));		
	memcpy(c_padded, matDiagUp, n * sizeof(double));
	memcpy(d_padded, rhs, n * sizeof(double));

	// copy values, and pad untill filled

	for (int i = n; i<NRC; i++){
		a_padded[i] = 0;
		b_padded[i] = 1; // helps solution be stable
		c_padded[i] = 0;
		d_padded[i] = 0;
	}

	if (debug){
		std::cout << "before:\nind\tA\tB\tC\tD\n";
		for (int i=0; i<NRC; i++){
			std::cout << i << "\t" << a_padded[i] << "\t" << b_padded[i] << "\t" << c_padded[i] << "\t" << d_padded[i] << "\n";
		}
	}

    xf::solver::gtsv<double, NRC, NCU>(NRC, a_padded, b_padded, c_padded, d_padded);

	if (debug){
		std::cout << "after:\nind\tA\tB\tC\tD\n";
		for (int i=0; i<NRC; i++){
			std::cout << i << "\t" << a_padded[i] << "\t" << b_padded[i] << "\t" << c_padded[i] << "\t" << d_padded[i] << "\n";
		}
	}

	memcpy(rhs, d_padded, n * sizeof(double)); // copy result back into rhs
};

void add4d(double* A, double* B, double* out, int* strides_offsets_out, int debug=0) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];
					O_val = A_val + B_val;

					if (debug){
						std::cout << "(i, j, k, l)" << i << ", " << j << ", " << k << ", " << l << ". A: " << A_val << " B: " << B_val << std::endl;   
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void sub4d(double* A, double* B, double* out, int* strides_offsets_out, int debug=0) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];
					O_val = A_val - B_val;

					if (debug){
						std::cout << "(i, j, k, l) " << i << ", " << j << ", " << k << ", " << l << " A_ind: " << A_ind << " A_val: " << A_val << " B_ind: " << B_ind << " B: " << B_val << " O_ind " << O_ind << " O val: " << O_val << std::endl;   
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void mult4d(double* A, double* B, double* out, int* strides_offsets_out, int debug=0) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];
					O_val = A_val * B_val;

					if (debug){
						std::cout << "(i, j, k, l) " << i << ", " << j << ", " << k << ", " << l 
						<< " A_ind: " << A_ind << " A_val: " << A_val 
						<< " B_ind: " << B_ind << " B: " << B_val 
						<< " O_ind " << O_ind << " O val: " << O_val << std::endl;   
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void div4d(double* A, double* B, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];
					O_val = A_val / B_val;

					out[O_ind] = O_val; 
				}
			}
		}
	}
}


void max4d(double* A, double* B, double* out, int* strides_offsets_out, int debug=0) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val > B_val){
						O_val = A_val;
					} else {
						O_val = B_val;
					}

					if (debug){
						std::cout << "(i, j, k, l) (" << i << ", " << j << ", " << k << ", " << l << "). A: " << A_val << " B: " << B_val << " out: " << O_val << std::endl;   
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void min4d(double* A, double* B, double* out, int* strides_offsets_out, int debug=0) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val > B_val){
						O_val = B_val;
					} else {
						O_val = A_val;
					}

					if (debug){
						std::cout << "(i, j, k, l) (" << i << ", " << j << ", " << k << ", " << l << "). A: " << A_val << " B: " << B_val << " out: " << O_val << std::endl;   
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}



void abs4d(double* A, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];

					if (A_val < 0){
						O_val = -A_val;
					} else {
						O_val = A_val;
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}


void sqrt4d(double* A, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];

					O_val = std::sqrt(A_val);
					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void not4d(double* A, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];

					if (A_val == 0){
						O_val = 1;
					} else
					{
						if (A_val == 1){
							O_val = 0;
						} 
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void get4d(double* A, double* B, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val >= B_val){
						O_val = 1;
					} else {
						O_val = 0;
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}


void gt4d(double* A, double* B, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val > B_val){
						O_val = 1;
					} else {
						O_val = 0;
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}


void eet4d(double* A, double* B, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val == B_val){
						O_val = 1;
					} else {
						O_val = 0;
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

void and4d(double* A, double* B, double* out, int* strides_offsets_out) {
	int A_offset[4];
	int B_offset[4];
	int out_offset[4];
	int A_stride[4];
	int B_stride[4];
	int out_stride[4];
	int A_scaled_stride[4];
	int B_scaled_stride[4];
	int out_scaled_stride[4];
	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[4 + i]);
		out_stride[i] = (strides_offsets_out[2*4 + i]);

		A_scaled_stride[i] = strides_offsets_out[i];
		B_scaled_stride[i] = strides_offsets_out[4 + i];
		out_scaled_stride[i] = strides_offsets_out[2*4 + i];

		A_offset[i] = (strides_offsets_out[3*4 + i]);
		B_offset[i] = (strides_offsets_out[4*4 + i]);
		out_offset[i] = (strides_offsets_out[5*4 + i]);

		out_shape[i] = (strides_offsets_out[6*4 +i]);
		out_end_offset[i] = (strides_offsets_out[7*4 + i]);
	}

	int A_lin_offset = (strides_offsets_out[8*4]); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*4 + 1]);
	int out_lin_offset = (strides_offsets_out[8*4 + 2]);

	int A_size = (strides_offsets_out[8*4 + 3]);
	int B_size = (strides_offsets_out[8*4 + 1 + 3]);
	int out_size = (strides_offsets_out[8*4 + 2 + 3]);

	int A_ind, B_ind, O_ind;
	double A_val, B_val, O_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3] + B_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					
					A_val = A[A_ind];
					B_val = B[B_ind];

					if (A_val == B_val){
						if (A_val == 1){
							O_val = 1;
						}
					} else {
						O_val = 0;
					}

					out[O_ind] = O_val; 
				}
			}
		}
	}
}

extern "C" void where4d(double* A, double* B, double* C, double* out, int* strides_offsets_out, int debug=0) {
	int O_ind, A_ind, B_ind, C_ind;
	double O_val, A_val, B_val, C_val;

	int A_offset[4];
	int B_offset[4];
	int C_offset[4];
	int out_offset[4];

	int A_stride[4];
	int B_stride[4];
	int C_stride[4];
	int out_stride[4];

	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<4; i++){
		A_stride[i] = strides_offsets_out[i];
		B_stride[i] = strides_offsets_out[4 + i];
		C_stride[i] = strides_offsets_out[2*4 + i];
		out_stride[i] = strides_offsets_out[3*4 + i];

		A_offset[i] = strides_offsets_out[4*4 + i];
		B_offset[i] = strides_offsets_out[5*4 + i];
		C_offset[i] = strides_offsets_out[6*4 + i];
		out_offset[i] = strides_offsets_out[7*4 + i];

		out_shape[i] = strides_offsets_out[8*4 +i];
		out_end_offset[i] = strides_offsets_out[9*4 + i];
	}

	int A_lin_offset = (strides_offsets_out[10*4]);
	int B_lin_offset = (strides_offsets_out[10*4 + 1]);
	int C_lin_offset = (strides_offsets_out[10*4 + 2]);
	int out_lin_offset = (strides_offsets_out[10*4 + 3]);

	int A_size = (strides_offsets_out[10*4 + 4]);
	int B_size = (strides_offsets_out[10*4 + 1 + 4]);
	int C_size = (strides_offsets_out[10*4 + 2 + 4]);
	int out_size = (strides_offsets_out[10*4 + 3 + 4]);

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=(0 + out_offset[2]); k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=(0 + out_offset[3]); l<(out_shape[3] + out_end_offset[3]); l++){

					A_ind = A_lin_offset + (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3];
					B_ind = B_lin_offset + (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + B_offset[3])*B_stride[3];
					C_ind = C_lin_offset + (i + C_offset[0])*C_stride[0] + (j + C_offset[1])*C_stride[1] + (k + C_offset[2])*C_stride[2] + (l + C_offset[3])*C_stride[3];

					O_ind = out_lin_offset + i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3];
					A_val = A[A_ind];
					B_val = B[B_ind];
					C_val = C[C_ind];

					if (A_val == 1){
						O_val = B_val;
					} else {
						if (A_val == 0){
							O_val = C_val;
						} 
					}

					if (debug){
						std::cout << "(i, j, k, l) " << i << ", " << j << ", " << k << ", " << l << " A_ind: " << A_ind << " A_val: " << A_val 
						<< " B_ind: " << B_ind << " B: " << B_val 
						<< " C_ind: " << C_ind << " C: " << C_val 
						<< " O_ind " << O_ind << " O val: " << O_val << std::endl;   
					}

					out[O_ind] = O_val;
				}
			}
		}
	}
}