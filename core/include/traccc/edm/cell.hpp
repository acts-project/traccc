/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"

// System include(s).
#include <cmath>
#include <limits>

namespace traccc {

/// Definition for one detector cell
///
/// It comes with two integer channel identifiers, an "activation value"
/// and a time stamp.
///
struct cell {
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;
};

/// Comparison / ordering operator for cells
TRACCC_HOST_DEVICE
inline bool operator<(const cell& lhs, const cell& rhs) {

    if (lhs.channel0 != rhs.channel0) {
        return (lhs.channel0 < rhs.channel0);
    } else if (lhs.channel1 != rhs.channel1) {
        return (lhs.channel1 < rhs.channel1);
    } else {
        return lhs.activation < rhs.activation;
    }
}

/// Equality operator for cells
TRACCC_HOST_DEVICE
inline bool operator==(const cell& lhs, const cell& rhs) {

    return (
        (lhs.channel0 == rhs.channel0) && (lhs.channel1 == rhs.channel1) &&
        (std::abs(lhs.activation - rhs.activation) < traccc::float_epsilon) &&
        (std::abs(lhs.time - rhs.time) < traccc::float_epsilon));
}

/// Header information for all of the cells in a specific detector module
///
/// It is handled separately from the list of all of the cells belonging to
/// the detector module, to be able to lay out the data in memory in a way
/// that is more friendly towards accelerators.
///
struct cell_module {

    event_id event = 0;
    geometry_id module = 0;
    transform3 placement = transform3{};

    channel_id range0[2] = {std::numeric_limits<channel_id>::max(), 0};
    channel_id range1[2] = {std::numeric_limits<channel_id>::max(), 0};

    pixel_data pixel{-8.425, -36.025, 0.05, 0.05};

};  // struct cell_module

// Declare all cell collection/container types
TRACCC_DECLARE_COLLECTION_TYPES(cell);
TRACCC_DECLARE_CONTAINER_TYPES(cell, cell_module, cell);

}  // namespace traccc
