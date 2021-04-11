/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "cuda/include/edm/cell_container.hpp"

namespace traccc {    

    struct label_container_cuda {
	vecmem::cuda::managed_memory_resource m_mem;	
	vecmem::jagged_vector< unsigned int > label;
	vecmem::vector< unsigned int > num_label;

	label_container_cuda(const cell_container_cuda& cells):
	    label(vecmem::jagged_vector< unsigned int >(&m_mem)),
	    num_label(vecmem::vector< unsigned int >(cells.items.size(),0,&m_mem))
	{
	    for(auto cell_module: cells.items){
		label.push_back(vecmem::vector< unsigned int >(cell_module.size(),0,&m_mem));
	    }
	}
	
    };
}
