/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"

namespace traccc {    

    struct measurement_container_cuda {
	event_id event = 0;
	vecmem::cuda::managed_memory_resource m_mem;	
	vecmem::vector <module_config> modcfg;
	vecmem::jagged_vector< measurement > items;

	measurement_container_cuda(const cell_container_cuda& cells, const label_container_cuda& labels):
	    modcfg(vecmem::vector<module_config>(&m_mem)),
	    items(vecmem::jagged_vector<measurement>(&m_mem))
	{
	    for(int i=0; labels.num_label.size(); ++i){
		modcfg = cells.modcfg;
		items.push_back(vecmem::vector< traccc::measurement >(labels.num_label[i], &m_mem));
	    }
	}
    }; 
}
