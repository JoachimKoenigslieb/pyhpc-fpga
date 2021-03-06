#include <cmath>
// https://github.com/Xilinx/HLS-Tiny-Tutorials/tree/master/algorithm_fixed_point_sqrt <cmath> should be implemented but slow!
extern "C" void sqrt4d(double* A, double* out, int A_lin_offset, int out_lin_offset, int* strides_offsets_out, int dim) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = plmem0 port = strides_offsets_out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = control port = A_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = out_lin_offset
#pragma HLS INTERFACE m_axi offset = slave bundle = control port = dim

	int O_ind, A_ind;
	double A_val;

	int A_offset[4];
	int out_offset[4];

	int A_stride[4];
	int out_stride[4];

	int out_end_offset[4];
	int out_shape[4];

	for (int i = 0; i<dim; i++){
		A_stride[i] = strides_offsets_out[i];
		out_stride[i] = strides_offsets_out[2*dim + i];

		A_offset[i] = strides_offsets_out[3*dim + i];
		out_offset[i] = strides_offsets_out[5*dim + i];

		out_shape[i] = strides_offsets_out[6*dim +i];
		out_end_offset[i] = strides_offsets_out[7*dim + i];
	}
	
	for (int i=out_offset[0]; i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=out_offset[1]; j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=out_offset[2]; k<(out_shape[2] + out_end_offset[2]); k++){
				for (int l=out_offset[3]; l<(out_shape[3] + out_end_offset[3]); l++){
					A_ind = (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + A_offset[3])*A_stride[3] + A_lin_offset;
					O_ind = i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + l*out_stride[3] + out_lin_offset;
					A_val = A[A_ind];
					out[O_ind] = std::sqrt(A_val);
				}
			}
		}
	}
}