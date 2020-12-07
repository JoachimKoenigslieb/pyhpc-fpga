extern "C" void gt2d(double* A, double* B, double* out, int A_lin_offset, int B_lin_offset, int out_lin_offset, int* strides_offsets_out, int dim) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = plmem0 port = strides_offsets_out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = control port = A_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = B_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = out_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = dim

	int O_ind, A_ind, B_ind;
	double A_val, B_val;

	int A_offset[dim];
	int B_offset[dim];
	int out_offset[dim];

	int A_stride[dim];
	int B_stride[dim];
	int out_stride[dim];

	int out_end_offset[dim];
	int out_shape[dim];

	for (int i = 0; i<dim; i++){
		A_stride[i] = strides_offsets_out[i];
		B_stride[i] = strides_offsets_out[dim + i];
		out_stride[i] = strides_offsets_out[2*dim + i];

		A_offset[i] = strides_offsets_out[3*dim + i];
		B_offset[i] = strides_offsets_out[4*dim + i];
		out_offset[i] = strides_offsets_out[5*dim + i];

		out_shape[i] = strides_offsets_out[6*dim +i];
		out_end_offset[i] = strides_offsets_out[7*dim + i];
	}
	
	for (int i=(0 + out_offset[0]); i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=(0 + out_offset[1]); j<(out_shape[1] + out_end_offset[1]); j++){
			A_ind = A_lin_offset + (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1];
			B_ind = B_lin_offset + (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1];
			O_ind = out_lin_offset + i*out_stride[0] + j*out_stride[1];
			A_val = A[A_ind];
			B_val = B[B_ind];
			if (A_val > B_val){
				out[O_ind] = 1;
			} else {
				out[O_ind] = 0;
			}

/*			std::cout << "(" << i << ", " << j <<  ")" << std::endl;
			std::cout << "\t\tO_ind: " << O_ind <<"\t\tO_val: " << out[O_ind] << std::endl;
			std::cout << "\t\tA_ind: " << A_ind <<"\t\tA_val: " << A_val << std::endl;
			std::cout << "\t\tB_ind: " << B_ind <<"\t\tB_val: " << B_val << std::endl; 
*/
		}
	}
}