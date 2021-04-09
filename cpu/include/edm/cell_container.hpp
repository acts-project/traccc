/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"
#include "edm/cell.hpp"
#include <vector>
#include <limits>

namespace traccc {
    /// A cell collection: 
    ///
    /// it remembers the moduleentifier and also 
    /// keeps track of the cell ranges for chosing optimal
    /// algorithm.
    struct cell_collection {
	event_id event = 0;
	module_config modcfg;
	std::vector< cell > items;
    };
    using cell_container = std::vector<cell_collection>;
}
