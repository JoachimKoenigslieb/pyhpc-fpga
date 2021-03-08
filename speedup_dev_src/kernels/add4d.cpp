#include <string.h>

#define X 6
#define Y 6
#define Z 4
#define SIZE X*Y*Z*3
#define TILE_SIZE 2

extern "C" void add4d(double* A, double* B, double* out, int* strides_offsets_out, int dim) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
#pragma HLS INTERFACE m_axi offset = slave bundle = plmem0 port = strides_offsets_out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = control port = dim

	int O_ind, A_ind, B_ind;
	double A_val, B_val;

	int A_offset[4];
	int B_offset[4];
	int out_offset[4];

	int A_stride[4];
	int B_stride[4];
	int out_stride[4];

	int out_end_offset[4];
	int out_shape[4];
	
	#pragma HLS DATAFLOW

	for (int i = 0; i<4; i++){
		#pragma HLS TRIPCOUNT max=4 min=1
		#pragma HLS UNROLL
		A_stride[i] = strides_offsets_out[i];
		B_stride[i] = strides_offsets_out[dim + i];
		out_stride[i] = strides_offsets_out[2*dim + i];

		A_offset[i] = strides_offsets_out[3*dim + i];
		B_offset[i] = strides_offsets_out[4*dim + i];
		out_offset[i] = strides_offsets_out[5*dim + i];

		out_shape[i] = strides_offsets_out[6*dim +i];
		out_end_offset[i] = strides_offsets_out[7*dim + i];
	}

	int A_lin_offset = strides_offsets_out[8*dim];
	int B_lin_offset = strides_offsets_out[8*dim + 1];
	int out_lin_offset = strides_offsets_out[8*dim + 2];

	int A_size = strides_offsets_out[8*dim + 3];
	int B_size = strides_offsets_out[8*dim + 1 + 3];
	int out_size = strides_offsets_out[8*dim + 2 + 3];

	double local_A[TILE_SIZE];
	double local_B[TILE_SIZE];
	double local_out[TILE_SIZE];

	int l_steps = out_shape[3] + out_end_offset[3];
	int l_tiles = l_steps / TILE_SIZE; // how many tiles we need to create

	/*
	std::cout << "creating " << l_tiles << " tiles of size " << TILE_SIZE << std::endl;
	std::cout << "A stride: " << A_stride[0] << ", " << A_stride[1] << ", " << A_stride[2] << ", " << A_stride[3] << std::endl;
	std::cout << "B stride: " << B_stride[0] << ", " << B_stride[1] << ", " << B_stride[2] << ", " << B_stride[3] << std::endl;
	std::cout << "out stride: " << out_stride[0] << ", " << out_stride[1] << ", " << out_stride[2] << ", " << out_stride[3] << std::endl;
	*/

	for (int i=out_offset[0]; i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=out_offset[1]; j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=out_offset[2]; k<(out_shape[2] + out_end_offset[2]); k++){
				// we tile innermost loop
				for (int l_tile=0; l_tile<l_tiles; l_tile++){
					memcpy(local_A, A + A_offset[3] * A_stride[3] + l_tile * TILE_SIZE * A_stride[3] + (i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + A_lin_offset, sizeof(double) * TILE_SIZE);
					memcpy(local_B, B + B_offset[3] * B_stride[3] + l_tile * TILE_SIZE * B_stride[3] + (i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + B_lin_offset, sizeof(double) * TILE_SIZE);

					for (int l=0; l<TILE_SIZE; l++){
						A_ind = l*A_stride[3];
						B_ind = l*B_stride[3];

						/*
						std::cout << "A should be: " 
						<< A[(i + A_offset[0])*A_stride[0] + (j + A_offset[1])*A_stride[1] + (k + A_offset[2])*A_stride[2] + (l + l_tile * TILE_SIZE + A_offset[3])*A_stride[3] + A_lin_offset]
						<< " but is " 
						<< local_A[A_ind] << "... l_tile is " << l_tile
						<< "... B should be " 
						<< B[(i + B_offset[0])*B_stride[0] + (j + B_offset[1])*B_stride[1] + (k + B_offset[2])*B_stride[2] + (l + l_tile * TILE_SIZE + B_offset[3])*B_stride[3] + B_lin_offset]
						<< " but is "
						<< local_B[B_ind] << std::endl;
						*/

						local_out[l] = local_A[A_ind] + local_B[B_ind];
					}
				
					memcpy(out + l_tile * TILE_SIZE + i*out_stride[0] + j*out_stride[1] + k*out_stride[2] + out_lin_offset, local_out, sizeof(double) * TILE_SIZE);
				}
			}
		}
	}
}