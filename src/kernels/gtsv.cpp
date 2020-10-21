#include "xf_solver_L2.hpp"


#define NRC 16
#define NCU 1

extern "C" void gtsv(int n, double* matDiagLow, double* matDiag, double* matDiagUp, double* rhs, double* out) {
#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = matDiagLow latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = matDiag latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = matDiagUp latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = rhs latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE m_axi offset = slave bundle = gmem3 port = out latency = 64 num_read_outstanding = \
    16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = matDiagLow bundle = control
#pragma HLS INTERFACE s_axilite port = matDiag bundle = control
#pragma HLS INTERFACE s_axilite port = matDiagUp bundle = control
#pragma HLS INTERFACE s_axilite port = rhs bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control


#pragma HLS INTERFACE s_axilite port = return bundle = control

    xf::solver::gtsv<double, NRC, NCU>(n, matDiagLow, matDiag, matDiagUp, rhs);
	out = rhs;
}