#include "../../cuda/src/utils/cuda_error_handling.hpp"
#include "white_paper_kernels.cuh"


__global__
void test(vecmem::data::jagged_vector_view< traccc::cell > _cells){
    
    vecmem::jagged_device_vector<traccc::cell> cells(_cells); 
    for (int i=0; i<cells.at(0).size(); i++){
	printf("%f \n",cells.at(0)[i].activation);
    }
}

__global__
void int_test_cuda(vecmem::data::jagged_vector_view< int > _data){
    
    vecmem::jagged_device_vector< int > data(_data); 
    for (int i=0; i<data.at(0).size(); i++){
	printf("%d \n",data.at(0)[i]);
    }
}


void cell_test(vecmem::data::jagged_vector_view< traccc::cell > cells)
{
    
    test<<< 1, cells.m_size >>>(cells);
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


void int_test(vecmem::data::jagged_vector_view< int > data){
    int_test_cuda<<< 1, data.m_size >>> (data);
    VECMEM_CUDA_ERROR_CHECK(cudaGetLastError());
    VECMEM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());    
}
