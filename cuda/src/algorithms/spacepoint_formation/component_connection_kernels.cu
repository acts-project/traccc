/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "cuda/include/algorithms/spacepoint_formation/component_connection_kernels.cuh"
#include "cuda/include/utils/cuda_error_check.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__device__
unsigned int find_root(const vecmem::device_vector< unsigned int >& L, unsigned int e){
  unsigned int r = e;
  while (L[r] != r){
    r = L[r];
  }
  return r;
}

__device__
unsigned int make_union(vecmem::device_vector< unsigned int >& L, unsigned int e1, unsigned int e2){
  int e;
  if (e1 < e2){
    e = e1;
    L[e2] = e;
  } else {
    e = e2;
    L[e1] = e;
  }
  return e;
}

__device__
bool is_adjacent(traccc::cell a, traccc::cell b){
  return (a.channel0 - b.channel0)*(a.channel0 - b.channel0) <= 1 
    and (a.channel1 - b.channel1)*(a.channel1 - b.channel1) <= 1; 
}

__device__
bool is_far_enough(traccc::cell a, traccc::cell b){
  return (a.channel1 - b.channel1) > 1; 
}

__device__
unsigned int sparse_ccl(const vecmem::device_vector< traccc::cell > cells,
			vecmem::device_vector< unsigned int >& L){    
    unsigned int start_j = 0;
    for (unsigned int i=0; i<cells.size(); ++i){
	L[i] = i;
	int ai = i;
	if (i > 0){
	    for (unsigned int j = start_j; j < i; ++j){
		if (is_adjacent(cells[i], cells[j])){
		    ai = make_union(L, ai, find_root(L, j));
		} else if (is_far_enough(cells[i], cells[j])){
		    ++start_j;
		}
	    }
	}    
    }
    
    // second scan: transitive closure
    unsigned int labels = 0;
    
    for (unsigned int i = 0; i < cells.size(); ++i){
	unsigned int l = 0;
	if (L[i] == i){
	    ++labels;
	    l = labels; 
	} else {
      l = L[L[i]];
	}
	L[i] = l;
    }
    
    return labels;
}


__global__
void sparse_ccl_kernel(vecmem::data::jagged_vector_view< traccc::cell > _cells_per_event,
		       vecmem::data::jagged_vector_view< unsigned int > _labels_per_event,
		       vecmem::data::vector_view<unsigned int> _num_labels);

namespace traccc {
    void sparse_ccl_cuda(traccc::cell_container_cuda& cells_per_event,
			 traccc::label_container_cuda& labels_per_event){
	
	vecmem::data::jagged_vector_data< cell > cell_data(cells_per_event.items,&cells_per_event.m_mem);
	vecmem::data::jagged_vector_data<unsigned int> label_data(labels_per_event.label,&labels_per_event.m_mem);
	auto num_labels = vecmem::get_data(labels_per_event.num_label);
	
	unsigned int num_threads = 64; 
	unsigned int num_blocks = cells_per_event.items.size()/num_threads + 1;
	
	sparse_ccl_kernel<<< num_blocks, num_threads >>>(cell_data,
							 label_data,
							 num_labels);        
	
	CUDA_ERROR_CHECK(cudaGetLastError());
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    }    
}

__global__
void sparse_ccl_kernel(vecmem::data::jagged_vector_view< traccc::cell > _cells_per_event,
		       vecmem::data::jagged_vector_view< unsigned int > _labels_per_event,
		       vecmem::data::vector_view<unsigned int> _num_labels){
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid>=_cells_per_event.m_size) return;
  
  vecmem::jagged_device_vector< traccc::cell > cells_per_event(_cells_per_event);
  vecmem::jagged_device_vector< unsigned int > labels_per_event(_labels_per_event);

  vecmem::device_vector< traccc::cell > cells_per_module = cells_per_event.at(gid);
  vecmem::device_vector< unsigned int > labels_per_module = labels_per_event.at(gid);
  vecmem::device_vector< unsigned int > num_labels(_num_labels);
  
  num_labels[gid] = sparse_ccl(cells_per_module, labels_per_module);
  
  return;
}
