/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"

namespace traccc {    

    struct spacepoint_container_cuda {
	event_id event = 0;
	geometry_id module = 0;
	vecmem::cuda::managed_memory_resource m_mem;	
	vecmem::jagged_vector< measurement > items;

	cell_event_cuda( void ):
	    items(vecmem::jagged_vector<spacepoint>(&m_mem))
	{
	}
    }; 
}
