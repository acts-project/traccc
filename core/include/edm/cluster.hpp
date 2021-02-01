/** TRACCC library, part of the ACTS project (R&D line)
 * 
 * (c) 2021 CERN for the benefit of the ACTS project
 * 
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"

#include <vector>
#include <functional>

namespace traccc {
    
    /// A cell definition: maximum two channel identifiers
    /// and one activiation value;
    struct cluster {
        cell_collection cells;
    };

    
    using signal_modeling = std::function<float(float)>;
    auto void_modeling = [](float signal_in) -> float {return signal_in; };

    struct cluster_collection {

        std::vector<cluster> items;
        // Next step information 
        float threshold = 0.;
        signal_modeling signal = void_modeling;
    };
}


