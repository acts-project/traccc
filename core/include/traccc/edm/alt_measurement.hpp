/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

namespace traccc {

/// Alternative measurement structure which contains both the measurement and
/// a link to its module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the
/// measurement container types.
struct alt_measurement {
    /// Local 2D coordinates for a measurement on a detector module
    point2 local{0., 0.};
    /// Variance on the 2D coordinates of the measurement
    variance2 variance{0., 0.};

    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link = 0;
};

/// Comparison / ordering operator for measurements
TRACCC_HOST_DEVICE
inline bool operator<(const alt_measurement& lhs, const alt_measurement& rhs) {

    if (lhs.module_link != rhs.module_link) {
        return lhs.module_link < rhs.module_link;
    } else if (std::abs(lhs.local[0] - rhs.local[0]) > float_epsilon) {
        return (lhs.local[0] < rhs.local[0]);
    } else {
        return (lhs.local[1] < rhs.local[1]);
    }
}

/// Equality operator for measurements
TRACCC_HOST_DEVICE
inline bool operator==(const alt_measurement& lhs, const alt_measurement& rhs) {

    return ((lhs.module_link == rhs.module_link) &&
            (std::abs(lhs.local[0] - rhs.local[0]) < float_epsilon) &&
            (std::abs(lhs.local[1] - rhs.local[1]) < float_epsilon) &&
            (std::abs(lhs.variance[0] - rhs.variance[0]) < float_epsilon) &&
            (std::abs(lhs.variance[1] - rhs.variance[1]) < float_epsilon));
}

/// Declare all alt measurement collection types
using alt_measurement_collection_types = collection_types<alt_measurement>;

}  // namespace traccc
