/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <vector>

namespace traccc {
    
    /// A cell definition: maximum two channel identifiers
    /// and one activiation value;
    struct spacepoint {
        point3 global = { 0., 0., 0.};
        covariance3 covariance = { 0., 0., 0.};
    };

    struct spacepoint_collection {     

        geometry_id module_id = 0;

        std::vector<spacepoint> items;
    };
}