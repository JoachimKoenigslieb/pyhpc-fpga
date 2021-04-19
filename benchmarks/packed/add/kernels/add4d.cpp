#include <ap_int.h>
#include <iostream>

#define TILE_SIZE 1
#define WIDTH 512
#define DATA_WIDTH 64
#define WORDS_IN_VAR (WIDTH / DATA_WIDTH)
#define MARS_WIDE_BUS_TYPE ap_uint<WIDTH>

#include "data_copyer.cpp"

void compute(double local_A[TILE_SIZE], double local_B[TILE_SIZE], double local_out[TILE_SIZE], int A_stride, int B_stride){
	#pragma HLS inline off // what tis does is a bit complicated. (i think it basicly pastes the hardware)
	int A_ind, B_ind; // free potimization: this int can be wayyyy smaller (4 bits actually, depends on TILE SIZE)
	
	for (int l=0; l<8; l++){
		#pragma HLS pipeline II=1
		A_ind = l*A_stride;
		B_ind = l*B_stride;

		local_out[l] = local_A[A_ind] + local_B[B_ind];
	}
}

void write(
	uint512_dt* out, double* local_out,
	int i, int j, int k, int l_tile,
	int* out_scaled_stride,
	int out_lin_offset)
{
	#pragma HLS inline off // what tis does is a bit complicated. (i think it basicly pastes the hardware)

	int out_ind_offset = l_tile + i*out_scaled_stride[0] + j*out_scaled_stride[1] + k*out_scaled_stride[2] + out_lin_offset;
	memcpy_wide_bus_write_double(out + out_ind_offset, local_out, 0, 8 * sizeof(double));
	return;
}

void read(
	uint512_dt* A, uint512_dt* B, 
	double* local_A, double* local_B, 
	int i, int j, int k, int l_tile, 
	int *A_scaled_stride, int *B_scaled_stride, 
	int* A_offset, int* B_offset, 
	int A_stride, int B_stride,
	int A_lin_offset, int B_lin_offset)
{
	#pragma HLS inline off
	int A_ind_offset = l_tile * A_stride + (i + A_offset[0])*A_scaled_stride[0] + (j + A_offset[1])*A_scaled_stride[1] + (k + A_offset[2])*A_scaled_stride[2] + A_lin_offset;
	int B_ind_offset = l_tile * B_stride + (i + B_offset[0])*B_scaled_stride[0] + (j + B_offset[1])*B_scaled_stride[1] + (k + B_offset[2])*B_scaled_stride[2] + B_lin_offset;

	memcpy_wide_bus_read_double(local_A, A + A_ind_offset, 0, 8 * sizeof(double));
	memcpy_wide_bus_read_double(local_B, B + B_ind_offset, 0, 8 * sizeof(double));
	return;
}

void add4d(uint512_dt* A, uint512_dt* B, uint512_dt* out, int* strides_offsets_out, int dim) {
	#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = A latency = 64 num_read_outstanding = \
		16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
	#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = B latency = 64 num_read_outstanding = \
		16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
	#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
		16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
	#pragma HLS INTERFACE m_axi offset = slave bundle = plmem0 port = strides_offsets_out latency = 64 num_read_outstanding = \
		16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

	#pragma HLS INTERFACE m_axi offset = slave bundle = control port = dim

	// array_partition creates invidual register for the memory. (does this mean if i dont do this, writing to local_A[0] will be blocking such that local_A[1] is not avaible to be written (or read)?)

	static int A_offset[4];
	static int B_offset[4];
	static int out_offset[4];
	static int A_stride[4];
	static int B_stride[4];
	static int out_stride[4];
	static int A_scaled_stride[4];
	static int B_scaled_stride[4];
	static int out_scaled_stride[4];
	static int out_end_offset[4];
	static int out_shape[4];

	#pragma HLS array_partition variable=A_stride
	#pragma HLS array_partition variable=B_stride
	#pragma HLS array_partition variable=out_stride

	#pragma HLS array_partition variable=A_offset
	#pragma HLS array_partition variable=B_offset

	for (int i = 0; i<4; i++){
		A_stride[i] = (strides_offsets_out[i]);
		B_stride[i] = (strides_offsets_out[dim + i]);
		out_stride[i] = (strides_offsets_out[2*dim + i]);

		A_scaled_stride[i] = strides_offsets_out[i] / 8;
		B_scaled_stride[i] = strides_offsets_out[dim + i] / 8;
		out_scaled_stride[i] = strides_offsets_out[2*dim + i] / 8;

		A_offset[i] = (strides_offsets_out[3*dim + i] / 8);
		B_offset[i] = (strides_offsets_out[4*dim + i] / 8);
		out_offset[i] = (strides_offsets_out[5*dim + i] / 8);

		out_shape[i] = (strides_offsets_out[6*dim +i]);
		out_end_offset[i] = (strides_offsets_out[7*dim + i] / 8);
	}

	int A_lin_offset = (strides_offsets_out[8*dim] / 8); // linear offsets should have a divmod 8 operation. (we need to keep any offsets around which are between data packing edges!)
	int B_lin_offset = (strides_offsets_out[8*dim + 1] / 8);
	int out_lin_offset = (strides_offsets_out[8*dim + 2] / 8);

	int A_size = (strides_offsets_out[8*dim + 3] / 8);
	int B_size = (strides_offsets_out[8*dim + 1 + 3] / 8);
	int out_size = (strides_offsets_out[8*dim + 2 + 3] / 8);

	double local_A[TILE_SIZE * 8];
	double local_B[TILE_SIZE * 8];
	double local_out[TILE_SIZE * 8];

	#pragma HLS array_partition variable=local_A
	#pragma HLS array_partition variable=local_B
	#pragma HLS array_partition variable=local_out

	int l_steps = out_shape[3] + out_end_offset[3];
	int l_tiles = l_steps / 8; // how many tiles we need to create

	uint512_dt A_elm, B_elm;

	for (int i=out_offset[0]; i<(out_shape[0] + out_end_offset[0]); i++){
		for (int j=out_offset[1]; j<(out_shape[1] + out_end_offset[1]); j++){
			for (int k=out_offset[2]; k<(out_shape[2] + out_end_offset[2]); k++){
				// we tile innermost loop
				for (int l_tile=0; l_tile<l_tiles; l_tile++){
					read(A, B, local_A, local_B, i, j, k, l_tile, A_scaled_stride, B_scaled_stride, A_offset, B_offset, A_stride[3], B_stride[3], A_lin_offset, B_lin_offset); 

					compute(local_A, local_B, local_out, A_stride[3], B_stride[3]);

					write(out, local_out, i, j, k, l_tile, out_scaled_stride, out_lin_offset);
				}
			}
		}
	}
}
}