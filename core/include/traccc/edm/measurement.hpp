/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// System include(s).
#include <limits>

namespace traccc {

/// Measurement structure which contains both the measurement and
/// a link to its module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the
/// measurement container types.
struct measurement {
    /// Local 2D coordinates for a measurement on a detector module
    point2 local{0., 0.};
    /// Variance on the 2D coordinates of the measurement
    variance2 variance{0., 0.};

    /// Geometry ID
    detray::geometry::barcode surface_link;

    /// Link to Module vector index
    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link = 0;

    /// Cluster link
    std::size_t cluster_link = std::numeric_limits<std::size_t>::max();
};

/// Comparison / ordering operator for measurements
TRACCC_HOST_DEVICE
inline bool operator<(const measurement& lhs, const measurement& rhs) {

    return lhs.surface_link < rhs.surface_link;
}

/// Equality operator for measurements
TRACCC_HOST_DEVICE
inline bool operator==(const measurement& lhs, const measurement& rhs) {

    return lhs.surface_link == rhs.surface_link;
}

/// Declare all measurement collection types
using measurement_collection_types = collection_types<measurement>;
/// Declare all measurement container types
using measurement_container_types = container_types<cell_module, measurement>;

}  // namespace traccc
