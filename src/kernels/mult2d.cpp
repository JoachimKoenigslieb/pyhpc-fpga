extern "C" void mult2d(double* A, double* B, int* A_stride, int* B_stride, int* A_offset, int* B_offset, double* out, int* out_shape, int* out_stride, int* out_offset, int* out_end_offset) {
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

	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			out[i*out_stride[0] + j*out_stride[1]] = A[(i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1]] * B[(i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1]];
		}
	}
}