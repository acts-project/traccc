/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "cuda/include/algorithms/spacepoint_formation/spacepoint_formation_kernels.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


namespace traccc{
/*
__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int> _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > > _geoInfo_per_event,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event);
*/
void sp_formation_cuda(const vecmem::data::jagged_vector_view< cell >& cell_per_event,
		       const vecmem::data::jagged_vector_view< unsigned int >& label_per_event,
		       const vecmem::data::vector_view< unsigned int >& num_labels,
		       const vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > >& geoInfo_per_event,
		       vecmem::data::jagged_vector_view< measurement > ms_per_event,
		       vecmem::data::jagged_vector_view< spacepoint> sp_per_event){
    unsigned int num_threads = 32;
    unsigned int num_blocks = cell_per_event.m_size/num_threads + 1;
    /*
    sp_formation_kernel<<< num_blocks, num_threads >>>(cell_per_event,
						       label_per_event,
						       num_labels,
						       geoInfo_per_event,
						       ms_per_event,
						       sp_per_event);    
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    */
  }  
/*
__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int > _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< std::tuple < traccc::geometry_id, traccc::transform3 > > _geoInfo_per_event,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event){
    int gid = blockDim.x * blockIdx.x + threadIdx.x; // module index
    if (gid>=_cell_per_event.m_size) return;
    
    vecmem::jagged_device_vector< cell > cell_per_event(_cell_per_event);
    vecmem::jagged_device_vector< unsigned int > label_per_event(_label_per_event);  
    vecmem::device_vector< unsigned int > num_labels(_num_labels);
    vecmem::device_vector< std::tuple < traccc::geometry_id, traccc::transform3 > > geoInfo_per_event(_geoInfo_per_event);
    vecmem::jagged_device_vector< measurement > ms_per_event(_ms_per_event);
    vecmem::jagged_device_vector< spacepoint > sp_per_event(_sp_per_event);  
    
    // retrieve cell and labels per module
    vecmem::device_vector< cell > cell_per_module = cell_per_event.at(gid);  
    vecmem::device_vector< unsigned int > label_per_module = label_per_event.at(gid);
    vecmem::device_vector< measurement > ms_per_module = ms_per_event.at(gid);
    vecmem::device_vector< spacepoint > sp_per_module = sp_per_event.at(gid);
    
    auto geoInfo = geoInfo_per_event[gid];
    auto labels = num_labels[gid];
    
    auto pix = traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05};
    
    for(int i=0; i<label_per_module.size(); ++i){
    unsigned int clabel = label_per_module[i]-1;
    scalar weight = cell_per_module[i].activation;
    
    auto cell_position = pix(cell_per_module[i].channel0, cell_per_module[i].channel1);
    auto& ms = ms_per_module[clabel];
    ms.weight_sum += weight;
    ms.local = ms.local + weight * cell_position;
    }
    
    for (int i_m = 0; i_m < ms_per_module.size(); ++i_m){
	auto& ms = ms_per_module[i_m];
	if( ms.weight_sum > 0 ) {
	    ms.local = 1./ms.weight_sum * ms.local;
	}
	point3 local_3d = {ms.local[0], ms.local[1], 0};
	auto& sp = sp_per_module[i_m];
	sp.global = std::get<1>(geoInfo).point_to_global(local_3d);
    } 
}
*/

}
