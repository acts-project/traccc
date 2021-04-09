/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/measurement.hpp"
#include "edm/cell.hpp"
#include <vector>

namespace traccc {

    struct measurement_collection {         
	event_id event = 0;
	module_config modcfg;
	std::vector<measurement> items;
    };
    using measurement_container = std::vector<measurement_collection>;
    
}
