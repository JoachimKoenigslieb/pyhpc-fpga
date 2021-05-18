typedef int inputType[4]; 

void sub_vecs(const int* A, const int* B, int* result){
	for (int i=0; i<4; i++){
		result[i] = A[i]-B[i];
	}
}


void add_vecs(const int* A, const int* B, int* result){
	for (int i=0; i<4; i++){
		result[i] = A[i]+B[i];
	}
}

void stride_from_shape(const int* A, int* stride, int debug = 0){
	stride[3] = 1;
	if (debug){
		//std::cout << "calculating the strides for shape:  [" << A[0] << ", " << A[1] << ", "  << A[2] << ", "  << A[3] << "]. \n";
		//std::cout << "i=" << "3 " << "stride: " << stride[0] << ", " << stride[1] << ", "  << stride[2] << ", "  << stride[3] << "\n";

	}
	for (int i=2; i>-1; i--){
		stride[i] = stride[i + 1] * A[i + 1];
		if (debug){
			//std::cout << "i=" << i << " stride: " << stride[0] << ", " << stride[1] << ", "  << stride[2] << ", "  << stride[3] << "\n";
		}

	}
}

void collect_linear_offset(const int* view_shape, const int* stride, const int* offset, int & lin_offset){
	lin_offset=0;

	for (int i=0; i<4; i++){
		if (view_shape[i] == 1)
			lin_offset += stride[i] * offset[i];
	}
}

void zero_on_squeeze(const int* view_shape, int* A){
	for (int i=0; i<4; i++){
		if (view_shape[i] != 1){
			continue;
		} else {
			A[i] = 0;
		}
	}
}

int cumprod(const int* v){
	int i = 1;
	for (int j=0; j<4; j++){
		i = i * v[j];
	}
	return i;
}

/*
void rebuild_stride(int* stride, int* view_shape){
	zero_on_squeeze(view_shape, stride);
}
*/


void copy_no_pad(int* to, const int* from){
	for (int i=0; i<4; i++){
		to[i] = from[i];
	}
}

void filter_on_squeeze(const int* view_shape, int* A, int initial_value, int debug=0){
	int new_A[4] = {initial_value, initial_value, initial_value, initial_value};
	int place_at = 3;
	for (int i=3; i>-1; i--){
		if (debug){
			//std::cout << "at i: " << i << ". view shape_i is " << view_shape[i];
		}
		if (view_shape[i] != 1){
			if (debug){
				//std::cout << "putting " << A[i] << " into position " << place_at << ". new_A currently looks like: " << new_A[0] << ", "  << new_A[1] << ", "  << new_A[2] << ", "  << new_A[3] << "\n";
			}
			new_A[place_at] = A[i];
			place_at -= 1; // fill from the back!
		} else
		{
			if (debug){
				//std::cout << "... continuing. new_A currently looks like: " << new_A[0] << ", "  << new_A[1] << ", "  << new_A[2] << ", "  << new_A[3] << "\n";
			}
		}	
	}

	if (debug){
		//std::cout << "\nfiltering...: " << debug << "\n";
		for (int i=0; i<4; i++){
			//std::cout << A[i] << ", ";
		}
		//std::cout << "became: \n";
		for (int i=0; i<4; i++){
			//std::cout << new_A[i] << ", ";
		}
		//std::cout << "\nwas that right?\n";
	}

	copy_no_pad(A, new_A);
}

void shift_right(int * array, int n){
	if (n > 0){
		for (int i=3; i>-1; i--){
			if (i<n){
				// less then n means we fill with zeros
				array[i] = 0;
			} else {
				array[i] = array[i-n];
			}
		}
	}
}

void rebuild_offset(int* offset, const int* view_shape, const int* out_offset, int out_dim, int dim, int debug=0){
	if (debug){
		//std::cout << "\nrebuilding offset\n... offset before zero on squeeze: ";
		//std::cout << offset[0] << ", "<< offset[1] << ", " << offset[2] << ", " << offset[3] << "\n";
	}
	zero_on_squeeze(view_shape, offset);
	if (debug){
		//std::cout << "offset before shifting right...: ";
		//std::cout << offset[0] << ", "<< offset[1] << ", " << offset[2] << ", " << offset[3] << "\n";
	}
	shift_right(offset, dim - out_dim);
	if (debug){
		//std::cout << "offset subtracting output offset (";
		//std::cout << out_offset[0] << ", "<< out_offset[1] << ", " << out_offset[2] << ", " << out_offset[3] << ") is: ";
		//std::cout << offset[0] << ", "<< offset[1] << ", " << offset[2] << ", " << offset[3] << "\n";
	}
	sub_vecs(offset, out_offset, offset);
	if (debug){
		//std::cout << "final offset is: ";
		//std::cout << offset[0] << ", "<< offset[1] << ", " << offset[2] << ", " << offset[3] << "\n";
	}
}


void copy_pad_back(int* to, const int* from, int pad_value, int A_dim){
	int offset = 4 - A_dim;

	for (int i=0; i<4; i++){
		//std::cout << "i is: " << i << " ... ";
		if (i<offset){
			////std::cout << "adding pad value!" << std::endl;
			to[i] = pad_value;
		} else {
			////std::cout << "adding from value..: " << from[i-offset] << std::endl;
			to[i] = from[i-offset];	
		}
	}
}

void negotiate_strides_fixed(
	const int * A_shape, 
	const int * out_shape, 
	const int * A_offset, 
	const int * out_offset,
	const int * A_end_offset, 
	const int * out_end_offset,
	int * A_stride, 
	int * out_stride,
	int * mutated_out_shape,
	int * mutated_out_offset,
	int * mutated_out_end_offset,
	int * mutated_A_offset, 
	int & A_lin_offset, 
	int & out_lin_offset, 
	int & A_data_size, 
	int & out_data_size,
	int out_dim,
	int A_dim,
	int debug
){

}

int count_squeezed(int * array){
	int count = 0;
	for (int i=0; i<4; i++){
		if (array[i] != 1){
			count += 1;
		}
	}
	return count;
}

void negotiate_strides(
	const int * A_shape, 
	const int * out_shape, 
	const int * A_offset, 
	const int * out_offset,
	const int * A_end_offset, 
	const int * out_end_offset,
	int * A_stride, 
	int * out_stride,
	int * mutated_out_shape,
	int * mutated_out_offset,
	int * mutated_out_end_offset,
	int * mutated_A_offset, 
	int & A_lin_offset, 
	int & out_lin_offset, 
	int & A_data_size, 
	int & out_data_size,
	int out_dim,
	int A_dim,
	int debug
){		
	int local_A_shape[4], local_A_end_offset[4], local_A_offset[4];
	int local_out_offset[4], local_out_end_offset[4], local_out_shape[4];

	copy_pad_back(local_A_end_offset, A_end_offset, 0, A_dim);
	copy_pad_back(local_A_offset, A_offset, 0, A_dim);
	copy_pad_back(local_A_shape, A_shape, 1, A_dim);

	copy_pad_back(local_out_offset, out_offset, 0, out_dim);
	copy_pad_back(local_out_end_offset, out_end_offset, 0, out_dim);
	copy_pad_back(local_out_shape, out_shape, 1, out_dim);

	int A_view_shape[4], out_view_shape[4];	
	add_vecs(local_A_shape, local_A_end_offset, A_view_shape);
	sub_vecs(A_view_shape, local_A_offset, A_view_shape);
	
	add_vecs(local_out_shape, local_out_end_offset, out_view_shape);
	sub_vecs(out_view_shape, local_out_offset, out_view_shape);

	int out_squeezed_dimension = count_squeezed(out_view_shape);

	if (debug){

	////std::cout << "local A shape: " << local_A_shape[0] << ", "  << local_A_shape[1] << ", "  << local_A_shape[2] << ", "  << local_A_shape[3] << ", " << std::endl;
	////std::cout << "local A offset: " << local_A_offset[0] << ", "  << local_A_offset[1] << ", "  << local_A_offset[2] << ", "  << local_A_offset[3] << ", " << std::endl;
	////std::cout << "local A end offset: " << local_A_end_offset[0] << ", "  << local_A_end_offset[1] << ", "  << local_A_end_offset[2] << ", "  << local_A_end_offset[3] << ", " << std::endl;
	////std::cout << "A view shape: " << A_view_shape[0] << ", "  << A_view_shape[1] << ", "  << A_view_shape[2] << ", "  << A_view_shape[3] << ", " << std::endl;
	//std::cout << "A dimension: " << A_dim << "\n";

	//std::cout << "\n";

	////std::cout << "out shape: " << local_out_shape[0] << ", "  << local_out_shape[1] << ", "  << local_out_shape[2] << ", "  << local_out_shape[3] << ", " << std::endl;
	////std::cout << "local out offset shape: " << local_out_offset[0] << ", "  << local_out_offset[1] << ", "  << local_out_offset[2] << ", "  << local_out_offset[3] << ", " << std::endl;
	////std::cout << "local out end offset shape: " << local_out_end_offset[0] << ", "  << local_out_end_offset[1] << ", "  << local_out_end_offset[2] << ", "  << local_out_end_offset[3] << ", " << std::endl;
	////std::cout << "out view shape: " << out_view_shape[0] << ", "  << out_view_shape[1] << ", "  << out_view_shape[2] << ", "  << out_view_shape[3] << ", " << std::endl;
	//std::cout << "out dimension: " << out_dim << ". out squeezed dim: " << out_squeezed_dimension << "\n\n";
	}
	
	stride_from_shape(local_A_shape, A_stride);
	stride_from_shape(local_out_shape, out_stride);

	if (debug){

	//std::cout << "\n";

	////std::cout << "A stride: " << A_stride[0] << ", "  << A_stride[1] << ", "  << A_stride[2] << ", "  << A_stride[3] << ", " << std::endl;
	////std::cout << "out stride: " << out_stride[0] << ", "  << out_stride[1] << ", "  << out_stride[2] << ", "  << out_stride[3] << ", " << std::endl;

	//std::cout << "\n";
	}
	
	collect_linear_offset(A_view_shape, A_stride, local_A_offset, A_lin_offset);
	collect_linear_offset(out_view_shape, out_stride, local_out_offset, out_lin_offset);

	zero_on_squeeze(A_view_shape, A_stride);
	zero_on_squeeze(out_view_shape, out_stride);

	int amount_right_shift = A_dim - out_squeezed_dimension;
	shift_right(A_stride, amount_right_shift);
	shift_right(out_stride, out_dim - out_squeezed_dimension);

	if (debug){
	////std::cout << "A stride (after squeeze/filter): " << A_stride[0] << ", "  << A_stride[1] << ", "  << A_stride[2] << ", "  << A_stride[3] << ", " << std::endl;
	////std::cout << "out stride (after squeeze/filter): " << out_stride[0] << ", "  << out_stride[1] << ", "  << out_stride[2] << ", "  << out_stride[3] << ", " << std::endl;
	}

	A_data_size = cumprod(local_A_shape);
	out_data_size = cumprod(local_out_shape);

	filter_on_squeeze(out_view_shape, local_out_offset, 0);
	filter_on_squeeze(out_view_shape, local_out_end_offset, 0);
	filter_on_squeeze(out_view_shape, local_out_shape, 1); 

	rebuild_offset(local_A_offset, A_view_shape, local_out_offset, out_squeezed_dimension, A_dim, debug);

	if (debug){
	////std::cout << "A offset (after rebuild): " << local_A_offset[0] << ", "  << local_A_offset[1] << ", "  << local_A_offset[2] << ", "  << local_A_offset[3] << ", " << std::endl;
	////std::cout << "out offset (after squeeze): " << local_out_offset[0] << ", "  << local_out_offset[1] << ", "  << local_out_offset[2] << ", "  << local_out_offset[3] << ", " << std::endl;
	////std::cout << "out end offset (after squeeze): " << local_out_end_offset[0] << ", "  << local_out_end_offset[1] << ", "  << local_out_end_offset[2] << ", "  << local_out_end_offset[3] << ", " << std::endl;
	}



	copy_no_pad(mutated_A_offset, local_A_offset);
	copy_no_pad(mutated_out_shape, local_out_shape);
	copy_no_pad(mutated_out_offset, local_out_offset);
	copy_no_pad(mutated_out_end_offset, local_out_end_offset);

	if (debug){

	////std::cout << "out view shape (after squeeze): " << local_out_shape[0] << ", "  << local_out_shape[1] << ", "  << local_out_shape[2] << ", "  << local_out_shape[3] << ", " << std::endl;
	////std::cout << "mutated stuf..:" << std::endl;
	////std::cout << "mutated A offset: " << mutated_A_offset[0] << ", "  << mutated_A_offset[1] << ", "  << mutated_A_offset[2] << ", "  << mutated_A_offset[3] << ", " << std::endl;
	////std::cout << "mutated out shape: " << mutated_out_shape[0] << ", "  << mutated_out_shape[1] << ", "  << mutated_out_shape[2] << ", "  << mutated_out_shape[3] << ", " << std::endl;
	////std::cout << "mutated out offset: " << mutated_out_offset[0] << ", "  << mutated_out_offset[1] << ", "  << mutated_out_offset[2] << ", "  << mutated_out_offset[3] << ", " << std::endl;
	////std::cout << "mutated out end offset: " << mutated_out_end_offset[0] << ", "  << mutated_out_end_offset[1] << ", "  << mutated_out_end_offset[2] << ", "  << mutated_out_end_offset[3] << ", " << std::endl;
	}
}

void calculate_strides(
	const inputType& A_shape,
	const inputType& B_shape,
	const inputType& output_shape,
	const inputType& A_offset,
	const inputType& B_offset,
	const inputType& output_offset,
	const inputType& A_offset_end,
	const inputType& B_offset_end,
	const inputType& output_offset_end,
	int stride_offsets_object[38],
	const int (&dimensions)[3],
	int debug=0
){
	int A_stride[4], B_stride[4], output_stride[4];
	int mutated_output_offset[4], mutated_output_end_offset[4], mutated_output_shape[4];
	int mutated_A_offset[4], mutated_B_offset[4];

	inputType output_offset_B_copy, output_offset_end_B_copy, output_stride_B_copy, output_shape_B_copy;
	copy_no_pad(output_offset_B_copy, output_offset);
	copy_no_pad(output_offset_end_B_copy, output_offset_end);
	copy_no_pad(output_stride_B_copy, output_stride);		
	copy_no_pad(output_shape_B_copy, output_shape);

	int A_lin_offset, B_lin_offset, output_lin_offset, A_data_size, B_data_size, output_data_size;

	negotiate_strides(
		A_shape, 
		output_shape, 
		A_offset,
		output_offset,
		A_offset_end,
		output_offset_end,
		A_stride,
		output_stride,
		mutated_output_shape,
		mutated_output_offset,
		mutated_output_end_offset,
		mutated_A_offset,
		A_lin_offset,
		output_lin_offset, 
		A_data_size, 
		output_data_size,
		dimensions[2],
		dimensions[0],
		debug);

	//std::cout << "after first (calc on A)... \n\n";

	////std::cout << "A offset: " << mutated_A_offset[0] << ", "  << mutated_A_offset[1] << ", "  << mutated_A_offset[2] << ", "  << mutated_A_offset[3] << ", " << std::endl;
	////std::cout << "out offset: " << mutated_output_offset[0] << ", "  << mutated_output_offset[1] << ", "  << mutated_output_offset[2] << ", "  << mutated_output_offset[3] << ", " << std::endl;

	negotiate_strides(
		B_shape, 
		output_shape_B_copy, 
		B_offset,
		output_offset_B_copy,
		B_offset_end,
		output_offset_end_B_copy,
		B_stride,
		output_stride_B_copy,
		mutated_output_shape,
		mutated_output_offset,
		mutated_output_end_offset,
		mutated_B_offset,
		B_lin_offset,
		output_lin_offset, 
		B_data_size, 
		output_data_size,
		dimensions[2],
		dimensions[1],
		debug);

	//std::cout << "after first (calc on B)... \n\n";

	////std::cout << "B offset: " << mutated_B_offset[0] << ", "  << mutated_B_offset[1] << ", "  << mutated_B_offset[2] << ", "  << mutated_B_offset[3] << ", " << std::endl;
	////std::cout << "out offset: " << mutated_output_offset[0] << ", "  << mutated_output_offset[1] << ", "  << mutated_output_offset[2] << ", "  << mutated_output_offset[3] << ", " << std::endl;


	for (int i = 0; i<4; i++){
		stride_offsets_object[i] = A_stride[i];
		stride_offsets_object[i + 1*4] = B_stride[i];
		stride_offsets_object[i + 2*4] = output_stride[i];

		stride_offsets_object[i + 3*4] = mutated_A_offset[i];
		stride_offsets_object[i + 4*4] = mutated_B_offset[i];
		stride_offsets_object[i + 5*4] = mutated_output_offset[i];

		stride_offsets_object[i + 6*4] = mutated_output_shape[i];
		stride_offsets_object[i + 7*4] = mutated_output_end_offset[i];
	}

	stride_offsets_object[8*4] = A_lin_offset;
	stride_offsets_object[8*4 + 1] = B_lin_offset;
	stride_offsets_object[8*4 + 2] = output_lin_offset;
	stride_offsets_object[8*4 + 3] = A_data_size;
	stride_offsets_object[8*4 + 3 + 1] = B_data_size;
	stride_offsets_object[8*4 + 3 + 2] = output_data_size;
}


void calculate_strides_where(
	const inputType& A_shape,
	const inputType& B_shape,
	const inputType& C_shape,
	const inputType& output_shape,
	const inputType& A_offset,
	const inputType& B_offset,
	const inputType& C_offset,
	const inputType& output_offset,
	const inputType& A_offset_end,
	const inputType& B_offset_end,
	const inputType& C_offset_end,
	const inputType& output_offset_end,
	int stride_offsets_object[48],
	const int (&dimensions)[4],
	int debug=0
){
	int A_stride[4], B_stride[4], C_stride[4], output_stride[4];
	int mutated_output_offset[4], mutated_output_end_offset[4], mutated_output_shape[4];
	int mutated_A_offset[4], mutated_C_offset[4], mutated_B_offset[4];

	inputType output_offset_B_copy, output_offset_end_B_copy, output_stride_B_copy, output_shape_B_copy;
	inputType output_offset_C_copy, output_offset_end_C_copy, output_stride_C_copy, output_shape_C_copy;

	copy_no_pad(output_offset_B_copy, output_offset);
	copy_no_pad(output_offset_end_B_copy, output_offset_end);
	copy_no_pad(output_stride_B_copy, output_stride);		
	copy_no_pad(output_shape_B_copy, output_shape);

	copy_no_pad(output_offset_C_copy, output_offset);
	copy_no_pad(output_offset_end_C_copy, output_offset_end);
	copy_no_pad(output_stride_C_copy, output_stride);		
	copy_no_pad(output_shape_C_copy, output_shape);

	int A_lin_offset, B_lin_offset, C_lin_offset, output_lin_offset, A_data_size, B_data_size, C_data_size, output_data_size;

	negotiate_strides(
		A_shape, 
		output_shape, 
		A_offset,
		output_offset,
		A_offset_end,
		output_offset_end,
		A_stride,
		output_stride,
		mutated_output_shape,
		mutated_output_offset,
		mutated_output_end_offset,
		mutated_A_offset,
		A_lin_offset,
		output_lin_offset, 
		A_data_size, 
		output_data_size,
		dimensions[3],
		dimensions[0],
		debug);

	negotiate_strides(
		B_shape, 
		output_shape_B_copy, 
		B_offset,
		output_offset_B_copy,
		B_offset_end,
		output_offset_end_B_copy,
		B_stride,
		output_stride_B_copy,
		mutated_output_shape,
		mutated_output_offset,
		mutated_output_end_offset,
		mutated_B_offset,
		B_lin_offset,
		output_lin_offset, 
		B_data_size, 
		output_data_size,
		dimensions[3],
		dimensions[1],
		debug);

	negotiate_strides(
		C_shape, 
		output_shape_C_copy, 
		C_offset,
		output_offset_C_copy,
		C_offset_end,
		output_offset_end_C_copy,
		C_stride,
		output_stride_C_copy,
		mutated_output_shape,
		mutated_output_offset,
		mutated_output_end_offset,
		mutated_C_offset,
		C_lin_offset,
		output_lin_offset, 
		C_data_size, 
		output_data_size,
		dimensions[3],
		dimensions[2],
		debug);

	//std::cout << "after first (calc on B)... \n\n";

	////std::cout << "B offset: " << mutated_B_offset[0] << ", "  << mutated_B_offset[1] << ", "  << mutated_B_offset[2] << ", "  << mutated_B_offset[3] << ", " << std::endl;
	////std::cout << "out offset: " << mutated_output_offset[0] << ", "  << mutated_output_offset[1] << ", "  << mutated_output_offset[2] << ", "  << mutated_output_offset[3] << ", " << std::endl;


	for (int i = 0; i<4; i++){
		stride_offsets_object[i] = A_stride[i];
		stride_offsets_object[i + 1*4] = B_stride[i];
		stride_offsets_object[i + 2*4] = C_stride[i];
		stride_offsets_object[i + 3*4] = output_stride[i];

		stride_offsets_object[i + 4*4] = mutated_A_offset[i];
		stride_offsets_object[i + 5*4] = mutated_B_offset[i];
		stride_offsets_object[i + 6*4] = mutated_C_offset[i];
		stride_offsets_object[i + 7*4] = mutated_output_offset[i];

		stride_offsets_object[i + 8*4] = mutated_output_shape[i];
		stride_offsets_object[i + 9*4] = mutated_output_end_offset[i];
	}

	stride_offsets_object[10*4] = A_lin_offset;
	stride_offsets_object[10*4 + 1] = B_lin_offset;
	stride_offsets_object[10*4 + 2] = C_lin_offset;
	stride_offsets_object[10*4 + 3] = output_lin_offset;

	stride_offsets_object[10*4 + 4] = A_data_size;
	stride_offsets_object[10*4 + 5] = B_data_size;
	stride_offsets_object[10*4 + 6] = C_data_size;
	stride_offsets_object[10*4 + 7] = output_data_size;
}


void print_stride_obj(int * obj){
	////std::cout << "\n\nPrinting stride obj..." << std::endl;
	////std::cout << "strides: " << std::endl;
	////std::cout << "A: " << obj[0] << ", " << obj[1] << ", " << obj[2] << ", " << obj[3] << ", " << std::endl;
	////std::cout << "B: " << obj[0 + 4] << ", " << obj[1 + 4] << ", " << obj[2 + 4] << ", " << obj[3 + 4] << ", " << std::endl;
	//////std::cout << "out: " << obj[0 + 8] << ", " << obj[1 + 8] << ", " << obj[2 + 8] << ", " << obj[3 + 8] << ", " << std::endl << std::endl;

	////std::cout << "offsets: " << std::endl;
	////std::cout << "A: " << obj[0 + 12] << ", " << obj[1 + 12] << ", " << obj[2 + 12] << ", " << obj[3 + 12] << ", " << std::endl;
	////std::cout << "B: " << obj[0 + 16] << ", " << obj[1 + 16] << ", " << obj[2 + 16] << ", " << obj[3 + 16] << ", " << std::endl;
	//////std::cout << "out: " << obj[0 + 20] << ", " << obj[1 + 20] << ", " << obj[2 + 20] << ", " << obj[3 + 20] << ", " << std::endl << std::endl;

	////std::cout << "output shape: " << std::endl;
	//////std::cout << "out: " << obj[0 + 24] << ", " << obj[1 + 24] << ", " << obj[2 + 24] << ", " << obj[3 + 24] << ", " << std::endl << std::endl;

	////std::cout << "output offset end: " << std::endl;
	//////std::cout << "out: " << obj[0 + 28] << ", " << obj[1 + 28] << ", " << obj[2 + 28] << ", " << obj[3 + 28] << ", " << std::endl << std::endl;

	////std::cout << "linear offsets: " << std::endl;
	////std::cout << "A: " << obj[32 + 0] << std::endl;
	////std::cout << "B: " << obj[32 + 1] << std::endl;
	//////std::cout << "out: " << obj[32 + 2] << std::endl << std::endl;;

	////std::cout << "data sizes: " << std::endl;
	////std::cout << "A: " << obj[32 + 3] << std::endl;
	////std::cout << "B: " << obj[32 + 4] << std::endl;
	//////std::cout << "out: " << obj[32 + 5] << std::endl << std::endl;;
}

