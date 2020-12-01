#include<iostream>
extern "C" void add1d(double* A, double* B, int* A_stride, int* B_stride, int* A_offset, int* B_offset, int A_lin_offset, int B_lin_offset, double* out, int* out_shape, int* out_stride, int* out_offset, int* out_end_offset, int out_lin_offset) {
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

	int A_ind, B_ind, O_ind;
	double A_val, B_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		O_ind = out_lin_offset + i*out_stride[0];
		A_ind = A_lin_offset + (i + A_offset[0])*A_stride[0];
		B_ind = B_lin_offset + (i + B_offset[0])*B_stride[0];

		A_val = A[A_ind];
		B_val = B[B_ind];
		out[O_ind] = A_val + B_val;

		//std::cout << "i,j: (" << i << ", " << j <<") O_ind: " << O_ind << " A_ind: " << A_ind << " B_ind: " << B_ind << " A_val: " << A_val << " B_val: " << B_val <<  " O_val: " << out[O_ind] << "\n";
	}
}