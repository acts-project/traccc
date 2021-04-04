/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "clusterization/component_connection_kernels.cuh"
#include "../../cuda/src/utils/cuda_error_handling.hpp" 
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace traccc{

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
bool is_adjacent(cell a, cell b){
  return (a.channel0 - b.channel0)*(a.channel0 - b.channel0) <= 1 
    and (a.channel1 - b.channel1)*(a.channel1 - b.channel1) <= 1; 
}

__device__
bool is_far_enough(cell a, cell b){
  return (a.channel1 - b.channel1) > 1; 
}

__device__
unsigned int sparse_ccl(const vecmem::device_vector< cell > cells,
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
void sparse_ccl_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
		     vecmem::data::jagged_vector_view< unsigned int > _label_per_event,
		     vecmem::data::vector_view<unsigned int> _num_labels);

__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int> _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > > _geoInfo_per_event,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event);

  void sparse_ccl_cuda(
		  const vecmem::data::jagged_vector_view< cell >& cell_per_event,
		  vecmem::data::jagged_vector_view< unsigned int> label_per_event,
		  vecmem::data::vector_view<unsigned int> num_labels){
    unsigned int num_threads = 32;
    unsigned int num_blocks = cell_per_event.m_size/num_threads + 1;
    sparse_ccl_kernel<<< num_blocks, num_threads >>>(cell_per_event,
						    label_per_event,
						    num_labels);        
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  }

  void sp_formation_cuda(const vecmem::data::jagged_vector_view< cell >& cell_per_event,
			 const vecmem::data::jagged_vector_view< unsigned int >& label_per_event,
			 const vecmem::data::vector_view< unsigned int >& num_labels,
			 const vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > >& geoInfo_per_event,
			 vecmem::data::jagged_vector_view< measurement > ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint> sp_per_event){
    unsigned int num_threads = 32;
    unsigned int num_blocks = cell_per_event.m_size/num_threads + 1;
    sp_formation_kernel<<< num_blocks, num_threads >>>(cell_per_event,
						       label_per_event,
						       num_labels,
						       geoInfo_per_event,
						       ms_per_event,
						       sp_per_event);    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  }  

__global__
void sparse_ccl_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
		       vecmem::data::jagged_vector_view< unsigned int > _label_per_event,
		       vecmem::data::vector_view<unsigned int> _num_labels){
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid>=_cell_per_event.m_size) return;
  
  vecmem::jagged_device_vector< cell > cell_per_event(_cell_per_event);
  vecmem::jagged_device_vector< unsigned int > label_per_event(_label_per_event);

  vecmem::device_vector< cell > cell_per_module = cell_per_event.at(gid);
  vecmem::device_vector< unsigned int > label_per_module = label_per_event.at(gid);
  vecmem::device_vector< unsigned int > num_labels(_num_labels);
  
  // run sparse_ccl
  num_labels[gid] = sparse_ccl(cell_per_module, label_per_module);

  //printf("%d", num_labels[gid]);
  
  return;
}


__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int > _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > > _geoInfo_per_event,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event){
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid>=_cell_per_event.m_size) return;
  
  vecmem::jagged_device_vector< cell > cell_per_event(_cell_per_event);
  vecmem::jagged_device_vector< unsigned int > label_per_event(_label_per_event);  
  vecmem::device_vector< unsigned int > num_labels(_num_labels);
  vecmem::jagged_device_vector< measurement > ms_per_event(_ms_per_event);
  vecmem::jagged_device_vector< spacepoint > sp_per_event(_sp_per_event);  

  // retrieve cell and labels per module
  vecmem::device_vector< cell > cell_per_module = cell_per_event.at(gid);  
  vecmem::device_vector< unsigned int > label_per_module = label_per_event.at(gid);
  vecmem::device_vector< measurement > ms_per_module = ms_per_event.at(gid);
  vecmem::device_vector< spacepoint > sp_per_module = sp_per_event.at(gid);  
  unsigned int labels = num_labels[gid];
  
  for(int i=0; i<label_per_module.size(); ++i){
    unsigned int clabel = label_per_module[i]-1;
    float weight = cell_per_module[i].activation;
    std::array<scalar, 2> cell_position({float(cell_per_module[i].channel0),
					 float(cell_per_module[i].channel1)});
    
    ms_per_module[clabel].weight_sum += weight;
    ms_per_module[clabel].local = ms_per_module[clabel].local + weight * cell_position;    
    //printf("%d %f \n", clabel, ms_per_module[clabel].total_weight);
  }

  for (auto ms: ms_per_module){
    if( ms.weight_sum > 0 ) ms.local = 1./ms.weight_sum * ms.local;

    //printf("%f %f \n", ms.local[0], ms.local[1]);
  }
  
}

}
