/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/algebra.hpp"
#include "definitions/primitives.hpp"
#include <vector>
#include <limits>
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/containers/jagged_vector.hpp"

namespace traccc {

    using channel_id = unsigned int;

    /// A cell definition: 
    ///
    /// maximum two channel identifiers
    /// and one activiation value, such as a time stamp
    struct cell {
        channel_id channel0 = 0;
        channel_id channel1 = 0;
        scalar activation = 0.;
        scalar time = 0.;
    };

    /// A module definition
    struct module_config{
	geometry_id module = 0;
	transform3 placement = transform3{};
	std::array<channel_id,2> range0 = {std::numeric_limits<channel_id>::max(), 0};
        std::array<channel_id,2> range1 = {std::numeric_limits<channel_id>::max(), 0};
    };
}

