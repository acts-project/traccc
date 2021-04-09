/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/spacepoint.hpp"
#include <vector>

namespace traccc {

    struct spacepoint_collection {     
    
	event_id event = 0;
	geometry_id module = 0;
	std::vector<spacepoint> items;
    };
    using spacepoint_container = std::vector<spacepoint_collection>;
    
}
