/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <climits>
#include <cmath>

namespace traccc {

/// Measurement expressed in local coordinates
///
/// It describes the 2D position and the uncertainty of that position
/// of a measurement on a detector element.
///
struct measurement {
    /// Local 2D coordinates for a measurement on a detector module
    point2 local{0., 0.};
    /// Variance on the 2D coordinates of the measurement
    variance2 variance{0., 0.};
    /// Cluster link
    std::size_t cluster_link = std::numeric_limits<std::size_t>::infinity();
};

/// Measurement with surface link
struct measurement_link {
    geometry_id surface_link;
    measurement meas;
};

/// Comparison / ordering operator for measurements
TRACCC_HOST_DEVICE
inline bool operator<(const measurement& lhs, const measurement& rhs) {

    if (std::abs(lhs.local[0] - rhs.local[0]) > float_epsilon) {
        return (lhs.local[0] < rhs.local[0]);
    } else {
        return (lhs.local[1] < rhs.local[1]);
    }
}

/// Equality operator for measurements
TRACCC_HOST_DEVICE
inline bool operator==(const measurement& lhs, const measurement& rhs) {

    return ((std::abs(lhs.local[0] - rhs.local[0]) < float_epsilon) &&
            (std::abs(lhs.local[1] - rhs.local[1]) < float_epsilon) &&
            (std::abs(lhs.variance[0] - rhs.variance[0]) < float_epsilon) &&
            (std::abs(lhs.variance[1] - rhs.variance[1]) < float_epsilon));
}

/// Declare all measurement collection types
using measurement_collection_types = collection_types<measurement>;
/// Declare all measurement container types
using measurement_container_types = container_types<cell_module, measurement>;

}  // namespace traccc
