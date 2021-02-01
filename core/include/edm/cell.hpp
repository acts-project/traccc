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
    /// A cell definition: maximum two channel identifiers
    /// and one activiation value;
    struct cell {
        unsigned int channel0 = 0;
        unsigned int channel1 = 0;
        float activiation = 0.;
    };

    using cell_collection = std::vector<cell>;
}

