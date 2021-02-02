/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "definitions/primitives.hpp"
#include <vector>

namespace traccc {
    
    /// A measurement definition, fix to two-dimensional here
    struct measurement {
        point2 local = {0., 0.};
        covariance2 covariance = { 0., 0.};
    };

    struct measurement_collection {     
        std::vector<measurement> items;
        // transform_index trf_index = 0;
    };
}