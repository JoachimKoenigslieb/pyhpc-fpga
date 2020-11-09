extern "C" void eet2d(double* A, double* B, int* A_stride, int* B_stride, int* A_offset, int* B_offset, double* out, int* out_shape, int* out_stride, int* out_offset, int* out_end_offset) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = A_stride bundle = control
#pragma HLS INTERFACE s_axilite port = A_offset bundle = control

#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = B_stride bundle = control
#pragma HLS INTERFACE s_axilite port = B_offset bundle = control

#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = out_stride bundle = control
#pragma HLS INTERFACE s_axilite port = out_offset bundle = control
#pragma HLS INTERFACE s_axilite port = out_end_offset bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control
	int O_ind, A_ind, B_ind;
	double A_val, B_val;

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1];
			B_ind = (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1];
			O_ind = i*out_stride[0] + j*out_stride[1];
			A_val = A[A_ind];
			B_val = B[B_ind];
			if (A_val == B_val){
				out[O_ind] = 1;
			} else {
				out[O_ind] = 0;
			}
/*
			std::cout << "(" << i << ", " << j << ", " << k << ", " << l << ")" << std::endl;
			std::cout << "\t\tO_ind: " << O_ind <<"\t\tO_val: " << out[O_ind] << std::endl;
			std::cout << "\t\tA_ind: " << A_ind <<"\t\tA_val: " << A_val << std::endl;
			std::cout << "\t\tB_ind: " << B_ind <<"\t\tB_val: " << B_val << std::endl; 
*/
		}
	}
}