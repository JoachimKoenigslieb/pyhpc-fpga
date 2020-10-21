extern "C" void mult3d(double* A, double* B, int* A_stride, int* B_stride, double* out, int* out_shape, int* out_stride) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = A_stride bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = B_stride bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control

#pragma HLS INTERFACE s_axilite port = return bundle = control

	for (int i=0; i<out_shape[0]; i++){
		for (int j=0; j<out_shape[1]; j++){
			for (int k=0; k<out_shape[2]; k++){
				out[i *out_stride[0] + j*out_stride[1] + k*out_stride[2]] = A[i*A_stride[0] + j*A_stride[1] + k*A_stride[2]] * B[i*B_stride[0] + j*B_stride[1] + k*B_stride[2]];
			}
		}
	}
}