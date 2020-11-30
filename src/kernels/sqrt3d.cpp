#include "hls_math.h"
extern "C" void sqrt3d(double* A, double* B, int* A_stride, int* B_stride, int* A_offset, int* B_offset, int A_lin_offset, int B_lin_offset, double* out, int* out_shape, int* out_stride, int* out_offset, int* out_end_offset, int out_lin_offset) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = A_stride bundle = control
#pragma HLS INTERFACE s_axilite port = A_offset bundle = control
#pragma HLS INTERFACE s_axilite port = A_lin_offset bundle = control

#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = B_stride bundle = control
#pragma HLS INTERFACE s_axilite port = B_offset bundle = control
#pragma HLS INTERFACE s_axilite port = B_lin_offset bundle = control

#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = out_stride bundle = control
#pragma HLS INTERFACE s_axilite port = out_offset bundle = control
#pragma HLS INTERFACE s_axilite port = out_end_offset bundle = control
#pragma HLS INTERFACE s_axilite port = out_lin_offset bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

	double val_A, val_B;

	for (int i=out_offset[0]; i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=out_offset[1]; j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=out_offset[2]; k<(out_shape[2] + out_end_offset[2]); k++){
				int A_ind, B_ind, O_ind;
				A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + A_lin_offset;
				O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + out_lin_offset;
				val_A = A[A_ind];
				out[O_ind] = hls::sqrt(val_A);
			}
		}
	}
}