/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#include "cuda/include/algorithms/spacepoint_formation/spacepoint_formation_kernels.cuh"
#include "cuda/include/utils/cuda_error_check.hpp"
#include "geometry/pixel_segmentation.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


namespace traccc{

__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int> _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< module_config > _modules,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event);

void sp_formation_cuda(traccc::cell_container_cuda& cells_per_event,
		       traccc::label_container_cuda& labels_per_event,
		       traccc::measurement_container_cuda& measurements_per_event,
		       traccc::spacepoint_container_cuda& spacepoints_per_event){
    unsigned int num_threads = 64;
    unsigned int num_blocks = cells_per_event.items.size()/num_threads + 1;
    
    vecmem::data::jagged_vector_data< cell > cell_data(cells_per_event.items,&cells_per_event.m_mem);
    vecmem::data::jagged_vector_data< unsigned int > label_data(labels_per_event.label,&labels_per_event.m_mem);
    auto num_labels = vecmem::get_data(labels_per_event.num_label);
    auto mod_data   = vecmem::get_data(measurements_per_event.modcfg);
    vecmem::data::jagged_vector_data< measurement > ms_data(measurements_per_event.items,&measurements_per_event.m_mem);
    vecmem::data::jagged_vector_data< spacepoint> sp_data(spacepoints_per_event.items,&spacepoints_per_event.m_mem);
        
    sp_formation_kernel<<< num_blocks, num_threads >>>(cell_data,
						       label_data,
						       num_labels,
						       mod_data,
						       ms_data,
						       sp_data);   
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    
  }  

__global__
void sp_formation_kernel(vecmem::data::jagged_vector_view< cell > _cell_per_event,
			 vecmem::data::jagged_vector_view< unsigned int> _label_per_event,
			 vecmem::data::vector_view< unsigned int> _num_labels,
			 vecmem::data::vector_view< module_config > _modules,
			 vecmem::data::jagged_vector_view< measurement > _ms_per_event,
			 vecmem::data::jagged_vector_view< spacepoint > _sp_per_event){
    int gid = blockDim.x * blockIdx.x + threadIdx.x; // module index
    if (gid>=_cell_per_event.m_size) return;
    
    vecmem::jagged_device_vector< cell > cell_per_event(_cell_per_event);
    vecmem::jagged_device_vector< unsigned int > label_per_event(_label_per_event);  
    vecmem::device_vector< unsigned int > num_labels(_num_labels);
    vecmem::device_vector< module_config > modules(_modules);
    vecmem::jagged_device_vector< measurement > ms_per_event(_ms_per_event);
    vecmem::jagged_device_vector< spacepoint > sp_per_event(_sp_per_event);  
    
    // retrieve cell and labels per module
    vecmem::device_vector< cell > cell_per_module = cell_per_event.at(gid);  
    vecmem::device_vector< unsigned int > label_per_module = label_per_event.at(gid);
    vecmem::device_vector< measurement > ms_per_module = ms_per_event.at(gid);
    vecmem::device_vector< spacepoint > sp_per_module = sp_per_event.at(gid);
    
    auto placement = modules[gid].placement;
    auto labels = num_labels[gid];
    
    auto pix = traccc::pixel_segmentation{-8.425, -36.025, 0.05, 0.05}; // Need to be removed
    
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
	//sp.global = std::get<1>(geoInfo).point_to_global(local_3d);
	sp.global = placement.point_to_global(local_3d);
    } 
}


}
