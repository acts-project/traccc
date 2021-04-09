/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"

namespace traccc {    

    struct cell_event_cuda {
	event_id event = 0;
	vecmem::cuda::managed_memory_resource m_mem;	
	vecmem::vector <module_config> modcfg;
	vecmem::jagged_vector< cell > items;

	cell_event_cuda( void ):
	    modules(vecmem::vector<module_config>(&m_mem)),
	    items(vecmem::jagged_vector<cell>(&m_mem))
	{
	}
    }; 
}
