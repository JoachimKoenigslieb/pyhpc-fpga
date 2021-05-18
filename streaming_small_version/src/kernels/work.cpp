#include "striding.cpp"
#include "operations.cpp"
//#include <algorithm>
#include <string.h>
#include <iostream>

#define X 6
#define Y 6
#define Z 6
#define SIZE 216

typedef int inputType[4]; 

extern "C" {
	void work(double arrays_1d[8 * X], double arrays_2d[2 * X * Y], double arrays_3d[6 * SIZE], double arrays_4d[5 * SIZE * 3], double out[SIZE * 3]){
		#pragma HLS INTERFACE m_axi offset = slave bundle = gmem0 port = arrays_1d latency = 64 num_read_outstanding = \
			16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
		#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = arrays_2d latency = 64 num_read_outstanding = \
			16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
		#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = arrays_3d latency = 64 num_read_outstanding = \
			16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16
		#pragma HLS INTERFACE m_axi offset = slave bundle = gmem1 port = arrays_4d latency = 64 num_read_outstanding = \
			16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

		#pragma HLS INTERFACE m_axi offset = slave bundle = gmem2 port = out latency = 64 num_read_outstanding = \
			16 num_write_outstanding = 16 max_read_burst_length = 64 max_write_burst_length = 64 depth = 16

		#pragma HLS DATAFLOW		

		int tau[1] = {0};
		double taup1[1] = {1.};
		double taum1[1] = {2.};
		double dt_tracer[1] = {1.};
		double dt_mom[1] = {1.};
		double dt_tke[1] = {1.};
		double AB_eps[1] = {0.1};
		double alpha_tke[1] = {1.};
		double c_eps[1] = {0.7};

		double one[1] = {1.};
		double zero[1] = {0.};
		double half[1] = {0.5};
		double two[1] = {2};
		double three_halfs[1] = {1.5};
		double K_h_tke[1] = {2000.};

		double tmp[SIZE];

		double dxt[X], dxu[X], dyt[X], dyu[X], dzt[X], dzw[X], cosu[X], cost[X];

		double Z_dim_arange[Z];

		for (int i=0; i<Z; i++){
			Z_dim_arange[i] = i;
		}

		for (int i=0; i<X; i++){
			dxt[i] = arrays_1d[i + 0];
			dxu[i] = arrays_1d[i + X];
			dyt[i] = arrays_1d[i + 2*X];
			dyu[i] = arrays_1d[i + 3*X];
			dzt[i] = arrays_1d[i + 4*X];
			dzw[i] = arrays_1d[i + 5*X];
			cosu[i] = arrays_1d[i + 6*X];
			cost[i] = arrays_1d[i + 7*X];
		}

		double kbot[X*Y], forc_tke_surface[X*Y];

		for (int i=0; i<(X * Y); i++){
			kbot[i] = arrays_2d[i];
			forc_tke_surface[i] = arrays_2d[i + (X * Y)];
		}

		double kappaM[SIZE], mxl[SIZE], forc[SIZE];
		double maskU[SIZE], maskV[SIZE], maskW[SIZE];

		for (int i=0; i<SIZE; i++){
			kappaM[i] = arrays_3d[i];
			mxl[i] = arrays_3d[i + SIZE];
			forc[i] = arrays_3d[i + 2*SIZE];
			maskU[i] = arrays_3d[i + 3*SIZE];
			maskV[i] = arrays_3d[i + 4*SIZE];
			maskW[i] = arrays_3d[i + 5*SIZE];
		}

		double u[SIZE * 3], v[SIZE * 3], w[SIZE * 3], tke[SIZE * 3], dtke[SIZE * 3];

		for (int i=0; i<(SIZE * 3); i++){
			u[i] = arrays_4d[i];
			v[i] = arrays_4d[i + (3*SIZE)];
			w[i] = arrays_4d[i + 2 * (3*SIZE)];
			tke[i] = arrays_4d[i + 3 * (3*SIZE)];
			dtke[i] = arrays_4d[i + 4 * (3*SIZE)];
		}

		double flux_east[X * Y * Z];
		double flux_north[X * Y * Z];
		double flux_top[X * Y * Z];
		double sqrttke[X * Y * Z];
		double a_tri[(X-2) * (Y-2) * Z];
		double b_tri[(X-2) * (Y-2) * Z];
		double b_tri_edge[(X-4) * (Y-4) * Z];
		double c_tri[(X-4) * (Y-4) * Z];
		double d_tri[(X-4) * (Y-4) * Z];
		double delta[(X-4) * (Y-4) * Z]; 
		double ks[(X-4) * (Y-4)];
		double tke_surf_corr[X * Y];

		int strides_kbot[38]; 
 
		calculate_strides(
			{X, Y}, {1}, {X-4, Y-4,},		//shapes
			{2, 2}, {0}, {0, 0},			//start index
			{-2, -2}, {0}, {0, 0}, 			//negativ
			strides_kbot, {2, 1, 2}
		);

		sub4d(kbot, one, ks, strides_kbot);

		int strides_sqrttke_1[38];

		calculate_strides(
			{X, Y, Z, 3}, {1}, {X, Y, Z,},				//shapes
			{0, 0, 0, 0}, {0}, {0, 0, 0,},				//start index
			{0, 0, 0, -2}, {0}, {0, 0, 0}, 				//negativ end index
			strides_sqrttke_1, {4, 1, 3}
		);

		max4d(tke, zero, sqrttke, strides_sqrttke_1);

		int strides_sqrttke_2[38]; 

		calculate_strides(
			{X, Y, Z,}, {1,}, {X, Y, Z,},						//shapes
			{0, 0, 0,}, {0,}, {0, 0, 0,},						//start index
			{0, 0, 0,}, {0,}, {0, 0, 0},
			strides_sqrttke_2, {3, 1, 3}	
		);

		sqrt4d(sqrttke, sqrttke, strides_sqrttke_2);

		// delta

		int strides_delta1[38];

		calculate_strides(
			{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},			//shapes
			{2, 2, 0}, {2, 2, 1}, {0, 0, 0,},		 	//start index
			{-2, -2, -1}, {-2, -2, 0}, {0, 0, -1}, 		//negativ end index
			strides_delta1, {3, 3, 3}
		);

		add4d(kappaM, kappaM, delta, strides_delta1);

		int strides_delta2[38];

		calculate_strides(
			{1,}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
			{0,}, {0, 0, 0}, {0, 0, 0,},				//start index
			{0,}, {0, 0, -1}, {0, 0, -1}, 				//negativ end index
			strides_delta2, {1, 3, 3}
		);

		mult4d(half, delta, delta, strides_delta2);

		int strides_delta3[38];

		calculate_strides(
			{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
			{0, 0, 0}, {1}, {0, 0, 0,},					//start index
			{0, 0, -1}, {0}, {0, 0, -1}, 				//negativ end index
			strides_delta3, {3, 1, 3}
		);

		div4d(delta, dzt, delta, strides_delta3);

		// a_tri

		int strides_a_tri1[38];

		calculate_strides(
			{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
			{0}, {0,0,0}, {0, 0, 1,},					//start index
			{0}, {0,0,-2}, {0, 0, -1}, 					//negativ end index
			strides_a_tri1, {1, 3, 3}		
		);

		sub4d(zero, delta, a_tri, strides_a_tri1);

		int strides_a_tri2[38];

		calculate_strides(
			{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
			{0, 0, 1}, {1}, {0, 0, 1,},					//start index
			{0, 0, -1}, {-1}, {0, 0, -1}, 				//negativ end index
			strides_a_tri2, {3, 1, 3}		
		);

		div4d(a_tri, dzw, a_tri, strides_a_tri2);

		// a_tri last index only

		int strides_a_tri3[38];

		calculate_strides(
			{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
			{0}, {0, 0, Z-2}, {0, 0, Z-1,},				//start index
			{0}, {0, 0, -1}, {0, 0, 0}, 				//negativ end index
			strides_a_tri3, {1, 3, 3}
		);

		sub4d(zero, delta, a_tri, strides_a_tri3);

		int strides_a_tri4[38];

		calculate_strides(
			{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
			{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
			{0, 0, 0}, {0}, {0, 0, 0}, 					//negativ end index
			strides_a_tri4, {3, 1, 3}
		);

		mult4d(a_tri, two, a_tri, strides_a_tri4);

		int strides_a_tri5[38];

		calculate_strides(
			{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
			{0, 0, Z-1}, {Z-1}, {0, 0, Z-1,},			//start index
			{0, 0, 0}, {0}, {0, 0, 0}, 					//negativ end index
			strides_a_tri5, {3, 1, 3}
		);

		div4d(a_tri, dzw, a_tri, strides_a_tri5);

		// b_tri
		int strides_b_tri1[38];
		double b_tri_temp[(X-4) * (Y-4) * Z];

		calculate_strides(
			{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
			{0, 0, 1}, {0, 0, 0}, {0, 0, 1,},			//start index
			{0, 0, -1}, {0, 0, -2}, {0, 0, -1},			//negativ end index
			strides_b_tri1, {3, 3, 3}
		);

		add4d(delta, delta, b_tri_temp, strides_b_tri1);

		int strides_b_tri2[38];

		calculate_strides(
			{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
			{0, 0, 1}, {1}, {0, 0, 1,},					//start index
			{0, 0, -1}, {-1}, {0, 0, -1}, 				//negativ end index
			strides_b_tri2, {3, 1, 3}
		);

		div4d(b_tri_temp, dzw, b_tri_temp, strides_b_tri2);

		int strides_b_tri3[38];

		calculate_strides(
			{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
			{0, 0, 1}, {0}, {0, 0, 1,},					//start index
			{0, 0, -1}, {0}, {0, 0, -1}, 				//negativ end index
			strides_b_tri3, {3, 1, 3}
		);

		add4d(b_tri_temp, one, b_tri, strides_b_tri3);

		// here we reuse tmp memory. that might be slow?
		int strides_b_tri4[38];

		calculate_strides(
			{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},		//shapes
			{2, 2, 1}, {2, 2, 1}, {0, 0, 1,},			//start index
			{-2, -2, -1}, {-2, -2, -1}, {0, 0, -1},		//negativ end index
			strides_b_tri4, {3, 3, 3}
		);

		div4d(sqrttke, mxl, b_tri_temp, strides_b_tri4);

		int strides_b_tri5[38];
			calculate_strides(
			{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
			{0, 0, 1}, {0}, {0, 0, 1,},					//start index
			{0, 0, -1}, {0}, {0, 0, -1}, 				//negativ end index
			strides_b_tri5, {3, 1, 3}
		);

		mult4d(b_tri_temp, c_eps, b_tri_temp, strides_b_tri5);

		int strides16[38];

		calculate_strides(
			{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
			{0, 0, 1}, {0, 0, 1}, {0, 0, 1,},			//start index
			{0, 0, -1}, {0, 0, -1}, {0, 0, -1},			//negativ end index
			strides16, {3, 3, 3}
		);

		add4d(b_tri, b_tri_temp, b_tri, strides16);
		// b_tri last index only!
		int strides17[38];

		calculate_strides(
			{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
			{0, 0, Z-2}, {Z-1}, {0, 0, Z-1},			//start index
			{0, 0, -1}, {0}, {0, 0, 0},					//negativ end index
			strides17, {3, 1, 3}
		);

		div4d(delta, dzw, b_tri_temp, strides17);

		int strides18[38];

		calculate_strides(
				{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		strides18, {3, 1, 3}	
		);

		mult4d(b_tri_temp, two, b_tri_temp, strides18);

		int strides19[38];

		calculate_strides(
			{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
			{0, 0, Z-1}, {0}, {0, 0, Z-1},				//start index
			{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
			strides19, {3, 1, 3}
		);

		add4d(b_tri_temp, one, b_tri, strides19);

		int strides20[38];

		calculate_strides(
				{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},		//shapes
		{2, 2, Z-1}, {2, 2, Z-1}, {0, 0, Z-1,},		//start index
		{-2, -2, 0}, {-2, -2, 0}, {0, 0, 0},		//negativ end index
		strides20, {3, 3, 3}	
		);

		div4d(sqrttke, mxl, b_tri_temp, strides20);

		int strides21[38];

		calculate_strides(
					{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1,},				//start index
		{0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
			strides21, {3, 1, 3}
		);

		mult4d(b_tri_temp, c_eps, b_tri_temp, strides21);

		int strides22[38];

		calculate_strides(
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, Z-1}, {0, 0, Z-1}, {0, 0, Z-1,},		//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},			//negativ end index
		strides22, {3, 3, 3}	
		);

		add4d(b_tri, b_tri_temp, b_tri, strides22);

		// b tri edge
		int strides23[38];

		calculate_strides(
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0}, {0, 0, 0},			//negativ end index
		strides23, {3, 1, 3}
		);

		div4d(delta, dzw, b_tri_edge, strides23);

		int strides24[38];

		calculate_strides(
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0}, {0, 0, 0},			//negativ end index
		strides24, {3, 1, 3}		
		);

		add4d(b_tri_edge, one, b_tri_edge, strides24);

		double b_tri_edge_tmp[(X-4) * (Y-4) * Z];

		int strides25[38];

		calculate_strides(
		{X, Y, Z}, {X, Y, Z}, {X-4, Y-4, Z},//shapes
		{2, 2, 0}, {2, 2, 0}, {0, 0, 0},		//start index
		{-2, -2, 0}, {-2, -2, 0}, {0, 0, 0},			//negativ end index
		strides25, {3, 3, 3}
		);

		div4d(sqrttke, mxl, b_tri_edge_tmp, strides25);

		mult4d(b_tri_edge_tmp, c_eps, b_tri_edge_tmp, strides24);

		int strides26[38];

		calculate_strides(
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},//shapes
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},		//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},			//negativ end index
		strides26, {3, 3, 3}
		);

		add4d(b_tri_edge_tmp, b_tri_edge, b_tri_edge, strides26);

		// c_tri

		int strides27[38];

		calculate_strides(
		{1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0}, {0, 0, 0}, {0, 0, 0,},					//start index
		{0}, {0, 0, -1}, {0, 0, -1},				//negativ end index
		strides27, {1, 3, 3}
		);

		sub4d(zero, delta, c_tri, strides27);

		int strides28[38];

		calculate_strides(
		{X-4, Y-4, Z}, {Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, 0}, {0}, {0, 0, 0,},					//start index
		{0, 0, -1}, {-1}, {0, 0, -1},				//negativ end index
		strides28, {3, 1, 3}
		);

		div4d(c_tri, dzw, c_tri, strides28);

		// d_tri

		int strides_dtri_1[38];

		calculate_strides(
		{X, Y, Z, 3}, {X, Y, Z,}, {X-4, Y-4, Z, },	//shapes
		{2, 2, 0, 0}, {2, 2, 0,}, {0, 0, 0,},		//start index
		{-2, -2, 0, -2}, {-2, -2, 0,}, {0, 0, 0,},	//negativ end index
		strides_dtri_1, {4, 3, 3}
		);

		add4d(tke, forc, d_tri, strides_dtri_1);

		int strides_dtri_2[38];

		double d_tri_tmp[(X-4)*(Y-4)*Z];

		calculate_strides(
		{X, Y,}, {Z}, {X-4, Y-4, Z} ,					//shapes
		{2, 2,}, {Z-1}, {0, 0, Z-1},					//start index
		{-2, -2,}, {0}, {0, 0, 0},						//negativ end index
		strides_dtri_2, {2, 1, 3}
		);

		div4d(forc_tke_surface, dzw, d_tri_tmp, strides_dtri_2);

		int strides_dtri_3[38];

		calculate_strides(
		{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		{0, 0, Z-1}, {0}, {0, 0, Z-1},					//start index
		{0, 0, 0}, {0}, {0, 0, 0},						//negativ end index
		strides_dtri_3, {3, 1, 3}
		);

		mult4d(d_tri_tmp, two, d_tri_tmp, strides_dtri_3);

		int strides_dtri_4[38];

		calculate_strides(
		{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},			//shapes
		{0, 0, Z-1}, {0, 0, Z-1}, {0, 0, Z-1},					//start index
		{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		strides_dtri_4, {3, 3, 3}
		); 

		add4d(d_tri_tmp, d_tri, d_tri, strides_dtri_4);

		//// Mask stuff

		//double land_mask[(X-4) * (Y-4)];
		//double edge_mask[(X-4) * (Y-4) * Z];
		//double water_mask[(X-4) * (Y-4) * Z];

		//int strides_land_mask[38];
   
		//calculate_strides(
		//	{X-4, Y-4}, {1}, {X-4, Y-4},					//shapes
		//	{0, 0}, {0}, {0, 0},					//start index
		//	{0, 0}, {0}, {0, 0},						//negativ end index
		//	strides_land_mask, {2, 1, 2}
		//);

		//get4d(ks, zero, land_mask, strides_land_mask);

		//int strides_edge_mask_1[38];

		//calculate_strides(
		//{1, 1, Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		//strides_edge_mask_1, {3, 3, 3}
		//);

		//eet4d(Z_dim_arange, ks, edge_mask, strides_edge_mask_1);

		//int strides_edge_mask_2[38];

		//calculate_strides(
		//{X-4, Y-4, 1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		//strides_edge_mask_2, {3, 3, 3}
		//);

		//and4d(land_mask, edge_mask, edge_mask, strides_edge_mask_2);

		//int strides_water_mask[38];

		//calculate_strides(
		//{1, 1, Z}, {X-4, Y-4, 1}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		//strides_water_mask, {3, 3, 3}
		//);

		//get4d(Z_dim_arange, ks, water_mask, strides_water_mask);

		//int strides_water_mask_2[38];

		//calculate_strides(
		//{X-4, Y-4, 1}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		//strides_water_mask_2, {3, 3, 3}
		//);

		//and4d(land_mask, water_mask, water_mask, strides_water_mask_2);

		//// Now apply masks

		//int strides_tri_masked[38];

		//calculate_strides(
		//{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0},						//negativ end index
		//strides_tri_masked, {3, 3, 3}
		//);

		//double not_edge_mask[(X-4) * (Y-4) * Z];

		//mult4d(water_mask, a_tri, a_tri, strides_tri_masked);
		//not4d(edge_mask, not_edge_mask, strides_tri_masked);
		//mult4d(a_tri, not_edge_mask, a_tri, strides_tri_masked);

		//int strides_b_tri_masked[48];

		//calculate_strides_where(
		//{X-4, Y-4, Z}, {X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0}, {0, 0, 0},					//negativ end index
		//strides_b_tri_masked, {3, 3, 1, 3}
		//);

		//where4d(water_mask, b_tri, one, b_tri, strides_b_tri_masked);

		//int strides_b_tri_masked_2[48];

		//calculate_strides_where(
		//{X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z}, {X-4, Y-4, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//start index
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},					//negativ end index
		//strides_b_tri_masked_2, {3, 3, 3, 3}
		//);

		//where4d(edge_mask, b_tri_edge, b_tri, b_tri, strides_b_tri_masked_2);

		//mult4d(water_mask, c_tri, c_tri, strides_tri_masked);

		//mult4d(water_mask, d_tri, d_tri, strides_tri_masked);

		//int strides_zero_hack_a_tri[38];

		//calculate_strides(
		//{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		//{0, 0, 0}, {0,}, {0, 0, 0},						//start index
		//{0, 0, -Z+1}, {0,}, {0, 0, -Z+1},						//negativ end index
		//strides_zero_hack_a_tri, {3, 1, 3}
		//);

		//mult4d(a_tri, zero, a_tri, strides_zero_hack_a_tri);

		//int strides_zero_hack_c_tri[38];

		//calculate_strides(
		//{X-4, Y-4, Z}, {1}, {X-4, Y-4, Z},			//shapes
		//{0, 0, Z-1}, {0,}, {0, 0, Z-1},						//start index
		//{0, 0, 0}, {0,}, {0, 0, 0},						//negativ end index
		//strides_zero_hack_c_tri, {3, 1, 3}
		//);

		//mult4d(c_tri, zero, c_tri, strides_zero_hack_c_tri);

		//gtsv((X-4) * (Y-4) * Z, a_tri, b_tri, c_tri, d_tri); // this gives buggy results if not padded to power of 2

		//int tke_strides[48];

		//calculate_strides_where(
		//	{X-4, Y-4, Z,}, {X-4, Y-4, Z,}, {X, Y, Z, 3}, {X, Y, Z, 3},
		//	{0, 0, 0}, {0, 0, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},
		//	{0, 0, 0}, {0, 0, 0}, {-2, -2, 0, -1}, {-2, -2, 0, -1},
		//	tke_strides, {3, 3, 4, 4}
		//);

		//where4d(water_mask, d_tri, tke, tke, tke_strides);

		//double mask[(X-4) * (Y-4)];
		//int mask_strides[38];

		//calculate_strides(
		//	{1}, {X, Y, Z, 3}, {X-4, Y-4,},					//shapes
		//	{0}, {2, 2, Z-1, 1}, {0, 0,},						//start index
		//	{0}, {-2, -2, 0, -1}, {0, 0, },					//negativ end index
		//	mask_strides, {1, 4, 2}
		//);

		//gt4d(zero, tke, mask, mask_strides);

		//int tke_surf_corr_strides_1[38];

		//calculate_strides(
		//	{1}, {X, Y, Z, 3}, {X, Y,},					//shapes
		//	{0}, {2, 2, Z-1, 1}, {2, 2,},						//start index
		//	{0}, {-2, -2, 0, -1}, {-2, -2, },					//negativ end index
		//	tke_surf_corr_strides_1, {1, 4, 2}
		//);

		//sub4d(zero, tke, tke_surf_corr, tke_surf_corr_strides_1);

		//int tke_surf_corr_strides_2[38];

		//calculate_strides(
		//	{1}, {X, Y, }, {X, Y,},					//shapes
		//	{0}, {2, 2,}, {2, 2,},						//start index
		//	{0}, {-2, -2}, {-2, -2, },					//negativ end index
		//	tke_surf_corr_strides_2, {1, 2, 2}
		//);

		//mult4d(half, tke_surf_corr, tke_surf_corr, tke_surf_corr_strides_2);

		//int tke_surf_corr_strides_3[38];

		//calculate_strides(
		//	{Z}, {X, Y, }, {X, Y,},					//shapes
		//	{Z-1}, {2, 2,}, {2, 2,},						//start index
		//	{0}, {-2, -2}, {-2, -2, },					//negativ end index
		//	tke_surf_corr_strides_3, {1, 2, 2}
		//);

		//mult4d(dzw, tke_surf_corr, tke_surf_corr, tke_surf_corr_strides_3);

		//int tke_surf_corr_masked_strides[48];

		//calculate_strides_where(
		//	{X-4, Y-4,}, {X, Y}, {1}, {X, Y,},
		//	{0, 0,}, {2, 2,}, {0,}, {2, 2,},
		//	{0, 0,}, {-2,-2,}, {0,}, {-2, -2,},
		//	tke_surf_corr_masked_strides, {2, 2, 1, 2}		
		//);

		//where4d(mask, tke_surf_corr, zero, tke_surf_corr, tke_surf_corr_masked_strides);

		//int tke_maximum_strides[38];
		
		//calculate_strides(
		//	{1}, {X, Y, Z, 3}, {X, Y, Z, 3},
		//	{0}, {2, 2, Z-1, 1}, {2, 2, Z-1, 1},
		//	{0}, {-2, -2, 0, -1}, {-2,-2, 0, -1},
		//	tke_maximum_strides, {1, 4, 4}
		//);

		//max4d(zero, tke, tke, tke_maximum_strides);

		//// flux east
		//int strides_flux_east_1[38];

		//calculate_strides(
		//	{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z},					//shapes
		//	{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,},						//start index
		//	{0, 0, 0, -2}, {-1, 0, 0, -2}, {-1, 0, 0, },					//negativ end index
		//	strides_flux_east_1, {4, 4, 3}
		//);

		//sub4d(tke, tke, flux_east, strides_flux_east_1);

		//int strides_flux_east_2[38];

		//calculate_strides(
		//	{1}, {X, Y, Z,}, {X, Y, Z},					//shapes
		//	{0,}, {0, 0, 0,}, {0, 0, 0,},						//start index
		//	{0,}, {-1, 0, 0, }, {-1, 0, 0, },					//negativ end index
		//	strides_flux_east_2, {1, 3, 3}
		//);

		//mult4d(K_h_tke, flux_east, flux_east, strides_flux_east_2);

		//int strides_flux_east_3[38];

		//calculate_strides(
		//	{X, Y, Z}, {1, X, 1}, {X, Y, Z},					//shapes
		//	{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		//	{-1, 0, 0}, { 0, 0, 0}, {-1, 0, 0, },					//negativ end index
		//	strides_flux_east_3, {3, 3, 3}
		//);

		//div4d(flux_east, cost, flux_east, strides_flux_east_3);

		//int strides_flux_east_4[38];

		//calculate_strides(
		//	{X, Y, Z}, {X, 1, 1}, {X, Y, Z},					//shapes
		//	{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		//	{-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0, },					//negativ end index
		//	strides_flux_east_4, {3, 3, 3}
		//);

		//div4d(flux_east, dxu, flux_east, strides_flux_east_4);

		//int strides_flux_east_5[38];

		//calculate_strides(
		//	{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//	{0, 0, 0,}, {0, 0, 0}, {0, 0, 0,},						//start index
		//	{-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0, },					//negativ end index
		//	strides_flux_east_5, {3, 3, 3}
		//);

		//mult4d(flux_east, maskU, flux_east, strides_flux_east_5);

		//int strides_flux_east_6[38];

		//calculate_strides(
		//	{X, Y, Z}, {1}, {X, Y, Z},					//shapes
		//	{X-1, 0, 0,}, {0,}, {X-1, 0, 0,},						//start index
		//	{0, 0, 0}, {0,}, {0, 0, 0, },					//negativ end index
		//	strides_flux_east_6, {3, 1, 3}
		//);

		//mult4d(flux_east, zero, flux_east, strides_flux_east_6);

		//// flux north
		//int strides_flux_north_1[38];

		//calculate_strides(
		//	{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z},					//shapes
		//	{0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0,},						//start index
		//	{0, 0, 0, -2}, {0, -1, 0, -2}, {0, -1, 0, },					//negativ end index
		//	strides_flux_north_1, {4, 4, 3}
		//);

		//sub4d(tke, tke, flux_north, strides_flux_north_1);

		//int strides_flux_north_2[38];

		//calculate_strides(
		//{1}, {X, Y, Z,}, {X, Y, Z},					//shapes
		//{0,}, {0, 0, 0,}, {0, 0, 0,},						//start index
		//{0,}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		//	strides_flux_north_2, {1, 3, 3}
		//);

		//mult4d(K_h_tke, flux_north, flux_north, strides_flux_north_2);

		//int strides_flux_north_3[38];

		//calculate_strides(
		//{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		//{0, 0, 0}, {0, 0, 0}, {0, 0, 0,},						//start index
		//{0, -1, 0}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		//	strides_flux_north_3, {3, 3, 3}
		//);

		//div4d(flux_north, dyu, flux_north, strides_flux_north_3);
		//mult4d(flux_north, cosu, flux_north, strides_flux_north_3);
		
		//int strides_flux_north_4[38];
		
		//calculate_strides(
		//	{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//	{0, 0, 0}, {0, 0, 0}, {0, 0, 0,},						//start index
		//	{0, -1, 0}, {0, -1, 0, }, {0, -1, 0, },					//negativ end index
		//	strides_flux_north_4, {3, 3, 3}
		//);
		
		//mult4d(flux_north, maskV, flux_north, strides_flux_north_4);

		//int strides_flux_north_5[38];

		//calculate_strides(
		//{X, Y, Z}, {1}, {X, Y, Z},					//shapes
		//{0, Y-1, 0,}, {0,}, {0, Y-1, 0,},						//start index
		//{0, 0, 0}, {0,}, {0, 0, 0, },					//negativ end index
		//	strides_flux_north_5, {3, 1, 3}
		//);

		//mult4d(flux_north, zero, flux_north, strides_flux_north_5);

		//// crazyyy tke stuff goes here

		//double tke_temp[X*Y*Z];

		//int strides_tke_1[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//{2, 2, 0,}, {1, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {-3, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_1, {3, 3, 3}
		//);

		//sub4d(flux_east, flux_east, tke_temp, strides_tke_1);

		//int strides_tke_2[38];

		//calculate_strides(
		//{X, Y, Z}, {X, 1, 1}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {2, 0, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {-2, 0, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_2, {3, 3, 3}
		//);

		//div4d(tke_temp, dxt, tke_temp, strides_tke_2);

		//int strides_tke_3[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {2, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {-2, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_3, {3, 3, 3}
		//);

		//mult4d(tke_temp, maskW, tke_temp, strides_tke_3);

		//int strides_tke_4[38];

		//calculate_strides(
		//{X, Y, Z}, {1, X, 1}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_4, {3, 3, 3}
		//);

		//div4d(tke_temp, cost, tke_temp, strides_tke_4);

		//int strides_tke_5[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},						//start index
		//{-2, -2, 0}, {-2, -2, 0, -1}, {-2, -2, 0, -1},					//negativ end index
		//strides_tke_5, {3, 4, 4}
		//);

		//add4d(tke_temp, tke, tke, strides_tke_5);

		//int strides_tke_6[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {2, 1, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {-2, -3, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_6, {3, 3, 3}
		//);

		//sub4d(flux_north, flux_north, tke_temp, strides_tke_6);

		//int strides_tke_7[38];

		//calculate_strides(
		//{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_7, {3, 3, 3}
		//);

		//div4d(tke_temp, cost, tke_temp, strides_tke_7);

		//int strides_tke_8[38];

		//calculate_strides(
		//{X, Y, Z}, {1, Y, 1}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {0, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {0, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_8, {3, 3, 3}
		//);

		//div4d(tke_temp, dyt, tke_temp, strides_tke_8);

		//int strides_tke_9[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z},					//shapes
		//{2, 2, 0}, {2, 2, 0,}, {2, 2, 0},						//start index
		//{-2, -2, 0}, {-2, -2, 0,}, {-2, -2, 0},					//negativ end index
		//strides_tke_9, {3, 3, 3}
		//);

		//mult4d(tke_temp, maskW, tke_temp, strides_tke_9);

		//int strides_tke_10[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0}, {2, 2, 0, 1}, {2, 2, 0, 1},						//start index
		//{-2, -2, 0}, {-2, 2, 0, -1}, {-2, -2, 0, -1},					//negativ end index
		//strides_tke_10, {3, 3, 3}
		//);

		//add4d(tke_temp, tke, tke, strides_tke_10);

		//// unrolling ad superbee

		//double maskUtr[X*Y*Z];

		//int strides_maskUtr[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z,}, {X, Y, Z,},					//shapes
		//{0, 0, 0}, {1, 0, 0,}, {0, 0, 0,},						//start index
		//{-1, 0, 0}, {0, 0, 0}, {-1, 0, 0},					//negativ end index
		//strides_maskUtr, {3, 3, 3}
		//);

		//mult4d(maskW, maskW, maskUtr, strides_maskUtr);

		//int strides_flux_east_assign[38];

		//calculate_strides(
		//{1}, {1}, {X, Y, Z,},					//shapes
		//{0,}, {0,}, {0, 0, 0,},						//start index
		//{0,}, {0,}, {0, 0, 0},					//negativ end index
		//strides_flux_east_assign, {1, 1, 3}
		//);

		//mult4d(zero, zero, flux_east, strides_flux_east_assign);

		//adv_superbee(u, tke, maskUtr, dxt, 0, cost, cosu,
		//			flux_east, {X, Y, Z}, {1, 2, 0}, {-2, -2, 0});

		//double maskVtr[X*Y*Z];

		//int strides_maskVtr[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z,}, {X, Y, Z,},					//shapes
		//{0, 1, 0}, {0, 0, 0,}, {0, 0, 0,},						//start index
		//{0, 0, 0}, {0, -1, 0}, {0, -1, 0},					//negativ end index
		//strides_maskVtr, {3, 3, 3}
		//);

		//mult4d(maskW, maskW, maskVtr, strides_maskVtr);

		//int strides_flux_north_assign[38];

		//calculate_strides(
		//{1}, {1}, {X, Y, Z,},					//shapes
		//{0,}, {0,}, {0, 0, 0,},						//start index
		//{0,}, {0,}, {0, 0, 0},					//negativ end index
		//strides_flux_north_assign, {1, 1, 3}
		//);

		//mult4d(zero, zero, flux_north, strides_flux_north_assign);

		//adv_superbee(v, tke, maskVtr, dyt, 1, cost, cosu, 
		//			flux_north, {X, Y, Z}, {2, 1, 0}, {-2, -2, -0});

		//double maskWtr[X*Y*Z];
	
		//int strides_maskWtr[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z,}, {X, Y, Z,},					//shapes
		//{0, 0, 1}, {0, 0, 0,}, {0, 0, 0,},						//start index
		//{0, 0, 0}, {0, 0, -1}, {0, 0, -1},					//negativ end index
		//strides_maskWtr, {3, 3, 3}
		//);

		//mult4d(maskW, maskW, maskWtr, strides_maskWtr);

		//int strides_flux_top_assign[38];

		//calculate_strides(
		//{1}, {1}, {X, Y, Z,},					//shapes
		//{0,}, {0,}, {0, 0, 0,},						//start index
		//{0,}, {0,}, {0, 0, 0},					//negativ end index
		//strides_flux_top_assign, {1, 1, 3}
		//);

		//mult4d(zero, zero, flux_top, strides_flux_top_assign);

		//adv_superbee(w, tke, maskWtr, dzw, 2, cost, cosu, 
		//			flux_top, {X, Y, Z}, {2, 2, 0}, {-2, -2, -1});

		//double dtke_temp[X * Y * Z * 3];

		//// dtke

		//int strides_dtke_1[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z, 3},					//shapes
		//{1, 2, 0,}, {2, 2, 0,}, {2, 2, 0, 0},						//start index
		//{-3, -2, 0,}, {-2, -2, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_1, {3, 3, 4}
		//);

		//sub4d(flux_east, flux_east, dtke, strides_dtke_1);

		//int strides_dtke_2[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {1, X, 1}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {0, 2, 0}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {0, -2, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_2, {4, 3, 4}
		//);

		//div4d(dtke, cost, dtke, strides_dtke_2);

		//int strides_dtke_3[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, 1, 1}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {2, 0, 0}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {-2, 0, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_3, {4, 3, 4}
		//);

		//div4d(dtke, dxt, dtke, strides_dtke_3);

		//int strides_dtke_4[38];

		//calculate_strides(
		//{X, Y, Z}, {X, Y, Z}, {X, Y, Z, 3},					//shapes
		//{2, 2 ,0,}, {2, 1, 0,}, {2, 2, 0, 0},						//start index
		//{-2, -2, 0}, {-2, -3, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_4, {3, 3, 4}
		//);

		//sub4d(flux_north, flux_north, dtke_temp, strides_dtke_4);

		//int strides_dtke_5[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {1, X, 1}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {0, 2, 0}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {0, -2, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_5, {4, 3, 4}
		//);

		//div4d(dtke_temp, cost, dtke_temp, strides_dtke_5);

		//int strides_dtke_6[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {1, Y, 1}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {0, 2, 0}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {0, -2, 0}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_6, {4, 3, 4}
		//);

		//div4d(dtke_temp, dyt, dtke_temp, strides_dtke_6);

		//int strides_dtke_7[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {2, 2, 0, 0}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {-2, -2, 0, -2}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_7, {4, 4, 4}
		//);

		//sub4d(dtke, dtke_temp, dtke, strides_dtke_7);

		//int strides_dtke_8[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z,}, {X, Y, Z, 3},					//shapes
		//{2, 2, 0, 0}, {2, 2, 0,}, {2, 2, 0, 0} ,						//start index
		//{-2, -2, 0, -2}, {-2, -2, 0,}, {-2, -2, 0, -2},					//negativ end index
		//strides_dtke_8, {4, 3, 4}
		//);

		//mult4d(dtke, maskW, dtke, strides_dtke_8);

		//int strides_dtke_9[38];

		//calculate_strides(
		//{X, Y, Z,}, {Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, 0,}, {0,}, {0, 0, 0, 0} ,						//start index
		//{0, 0, -Z+1,}, {-Z+1,}, {0, 0, -Z+1, -2},					//negativ end index
		//strides_dtke_9, {3, 1, 4}
		//);

		//div4d(flux_top, dzw, dtke_temp, strides_dtke_9);

		//int strides_dtke_10[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{0, 0, 0, 0}, {0, 0, 0 ,0}, {0, 0, 0, 0} ,						//start index
		//{0, 0, -Z+1, -2}, {0, 0, -Z+1, -2}, {0, 0, -Z+1, -2},					//negativ end index
		//strides_dtke_10, {4, 4, 4}
		//);

		//sub4d(dtke, dtke_temp, dtke, strides_dtke_10);

		//int strides_dtke_11[38];

		//calculate_strides(
		//{X, Y, Z,}, {X, Y, Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, 0,}, {0, 0, 1}, {0, 0, 1, 0} ,						//start index
		//{0, 0, -2,}, {0, 0, -1,}, {0, 0, -1, -2},					//negativ end index
		//strides_dtke_11, {3, 3, 4}
		//);

		//sub4d(flux_top, flux_top, dtke_temp, strides_dtke_11);

		//int strides_dtke_12[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, 1, 0,}, {1}, {0, 0, 1, 0} ,						//start index
		//{0, 0, -1, -2,}, {-1,}, {0, 0, -1, -2},					//negativ end index
		//strides_dtke_12, {4, 1, 4}
		//);

		//div4d(dtke_temp, dzw, dtke_temp, strides_dtke_12);

		//int strides_dtke_13[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{0, 0, 1, 0}, {0, 0, 1 ,0}, {0, 0, 1, 0} ,						//start index
		//{0, 0, -1, -2}, {0, 0, -1, -2}, {0, 0, -1, -2},					//negativ end index
		//strides_dtke_13, {4, 4, 4}
		//);

		//add4d(dtke, dtke_temp, dtke, strides_dtke_13);

		//int strides_dtke_14[38];

		//calculate_strides(
		//{X, Y, Z,}, {X, Y, Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, Z-2,}, {0, 0, Z-1}, {0, 0, Z-1, 0} ,						//start index
		//{0, 0, -1,}, {0, 0, 0,}, {0, 0, 0, -2},					//negativ end index
		//strides_dtke_14, {3, 3, 4}
		//);

		//sub4d(flux_top, flux_top, dtke_temp, strides_dtke_14);

		//int strides_dtke_15[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, Z-1, 0,}, {Z-1}, {0, 0, Z-1, 0} ,						//start index
		//{0, 0, 0, -2,}, {0, }, {0, 0, 0, -2},					//negativ end index
		//strides_dtke_15, {4, 1, 4}
		//);

		//div4d(dtke_temp, dzw, dtke_temp, strides_dtke_15);

		//int strides_dtke_16[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {1,}, {X, Y, Z, 3},					//shapes
		//{0, 0, Z-1, 0,}, {0}, {0, 0, Z-1, 0} ,						//start index
		//{0, 0, 0, -2,}, {0, }, {0, 0, 0, -2},					//negativ end index
		//strides_dtke_16, {4, 1, 4}
		//);

		//mult4d(dtke_temp, two, dtke_temp, strides_dtke_16);

		//int strides_dtke_17[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{0, 0, Z-1, 0}, {0, 0, Z-1 ,0}, {0, 0, Z-1, 0} ,						//start index
		//{0, 0, 0, -2}, {0, 0, 0, -2}, {0, 0, 0, -2},					//negativ end index
		//strides_dtke_17, {4, 4, 4}
		//);

		//add4d(dtke, dtke_temp, dtke, strides_dtke_17);

		//// tke

		//double three_halves_plus_Ab_eps[1] = {1.5 + 0.1}; 
		//double one_halves_plus_Ab_eps[1] = {0.5 + 0.1}; 

		//double tke_temp2[X*Y*Z];

		//int strides_tke_temp2_1[38];

		//calculate_strides(
		//{1,}, {X, Y, Z, 3}, {X, Y, Z,},					//shapes
		//{0,}, {0, 0, 0, 0}, {0, 0, 0,} ,						//start index
		//{0,}, {0, 0, 0, -2}, {0, 0, 0,},					//negativ end index
		//strides_tke_temp2_1, {1, 4, 3}
		//);

		//mult4d(three_halves_plus_Ab_eps, dtke, tke_temp2, strides_tke_temp2_1);

		//int strides_tke_temp2_2[38];

		//calculate_strides(
		//{X, Y, Z,}, {X, Y, Z, 3}, {X, Y, Z, 3},					//shapes
		//{0, 0, 0,}, {0, 0, 0, 1}, {0, 0, 0, 1} ,						//start index
		//{0, 0, 0,}, {0, 0, 0, -1}, {0, 0, 0, -1},					//negativ end index
		//strides_tke_temp2_2, {3, 4, 4}
		//);

		//add4d(tke_temp2, tke, tke, strides_tke_temp2_2);

		//int strides_tke_temp2_3[38];

		//calculate_strides(
		//{1,}, {X, Y, Z, 3}, {X, Y, Z,},					//shapes
		//{0,}, {0, 0, 0, 2}, {0, 0, 0,} ,						//start index
		//{0,}, {0, 0, 0, 0}, {0, 0, 0},					//negativ end index
		//strides_tke_temp2_3, {1, 4, 3}
		//);

		//mult4d(one_halves_plus_Ab_eps, dtke, tke_temp2, strides_tke_temp2_3);

		//int strides_tke_temp2_4[38];

		//calculate_strides(
		//{X, Y, Z, 3}, {X, Y, Z,}, {X, Y, Z, 3},					//shapes
		//{0, 0, 0, 1}, {0, 0, 0}, {0, 0, 0, 1} ,						//start index
		//{0, 0, 0, -1}, {0, 0, 0}, {0, 0, 0, -1},					//negativ end index
		//strides_tke_temp2_4, {4, 3, 4}
		//);

		//sub4d(tke, tke_temp2, tke, strides_tke_temp2_4);

		//memcpy(out, tke, SIZE*3 * sizeof(double));
	}
}
