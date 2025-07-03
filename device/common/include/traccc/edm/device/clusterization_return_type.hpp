/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"

// System include(s).
#include <optional>

namespace traccc::device {

/// Return type for the device clusterization algorithms
struct clusterization_return_type {

    /// Measurements reconstructed by the clusterization algorithm
    measurement_collection_types::buffer measurements;

    /// Silicon clusters (optionally) reconstructed by the clusterization
    /// algorithm
    std::optional<edm::silicon_cluster_collection::buffer> clusters;

};  // struct clusterization_return_type

}  // namespace traccc::device
