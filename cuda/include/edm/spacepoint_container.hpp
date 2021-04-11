/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"

namespace traccc {    

    struct spacepoint_container_cuda {
	event_id event = 0;
	vecmem::cuda::managed_memory_resource m_mem;
	vecmem::vector< geometry_id > module;
	vecmem::jagged_vector< spacepoint > items;

	spacepoint_container_cuda(const cell_container_cuda& cells, const label_container_cuda& labels):
	    items(vecmem::jagged_vector<spacepoint>(&m_mem))
	{
	    for(int i=0; labels.num_label.size(); ++i){
		module.push_back(cells.modcfg[i].module);
		items.push_back(vecmem::vector< traccc::spacepoint >(labels.num_label[i], &m_mem));
	    }

	}
    }; 
}
