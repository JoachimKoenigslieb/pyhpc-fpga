#include <vector>
#include <fstream>
#include "cnpy.h"
#include <iostream>
#include <map>

#define DATA_SIZE 8
#define INDEXER(i, j) i * DATA_SIZE + j

template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

void m_softwareGold(
                    std::vector<int,aligned_allocator<int>> &in1,   //Input Matrix 1
                    std::vector<int,aligned_allocator<int>> &in2,   //Input Matrix 2
                    std::vector<int,aligned_allocator<int>> &out    //Output Matrix
                   )
{
    //Perform Matrix multiply Out = In1 x In2
    for(int i = 0; i < DATA_SIZE; i++) {
        for(int j = 0; j < DATA_SIZE; j++) {
            for(int k = 0; k < DATA_SIZE; k++) {
                out[i * DATA_SIZE + j] += in1[i * DATA_SIZE + k] * in2[k * DATA_SIZE + j];
            }
        }
    }
}  

void set_input_matrix(std::vector<int, aligned_allocator<int>> &in, int* &data){
	for (int i=0; i<DATA_SIZE; i++){
		for (int j=0; j<DATA_SIZE; j++){
			in[INDEXER(i, j)] = data[INDEXER(i, j)];
		}
	};
}

void debug_print_matrix(std::vector<int, aligned_allocator<int>> &mat){ //usefull for quickly checking sizes, alignments etc.
	assert(mat.size() == DATA_SIZE * DATA_SIZE);

	for (int i=0; i<DATA_SIZE; i++){
		for (int j=0; j<DATA_SIZE; j++){
			std::cout <<  mat[INDEXER(i, j)] << ", ";
		}
		std::cout << "\n";
	};


}

int main(int argc, char** argv){
	int mat_size = DATA_SIZE * DATA_SIZE;
	std::vector<int, aligned_allocator<int>> in(mat_size);

	cnpy::npz_t np_arrs = cnpy::npz_load("mat_files.npz");

	cnpy::NpyArray np_A = np_arrs["upper"];
	cnpy::NpyArray np_B = np_arrs["diag"];
	cnpy::NpyArray np_res = np_arrs["lower"];

	int* A_data = np_A.data<int>();
	int* B_data = np_B.data<int>();
	int* res_data = np_res.data<int>();

	return 0;
}