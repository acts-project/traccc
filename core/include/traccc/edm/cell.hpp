/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"

namespace traccc {

/// Definition of a detector module
///
/// It is handled separately from the list of all of the cells belonging to
/// the detector module, to be able to lay out the data in memory in a way
/// that is more friendly towards accelerators.
///
struct cell_module {

    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0;

    pixel_data pixel;

};  // struct cell_module

/// Declare all cell module collection types
using cell_module_collection_types = collection_types<cell_module>;

/// Equality operator for cell module
TRACCC_HOST_DEVICE
inline bool operator==(const cell_module& lhs, const cell_module& rhs) {
    return lhs.module == rhs.module;
}

/// Definition for one detector cell
///
/// It comes with two integer channel identifiers, an "activation value",
/// a time stamp and a link to its module (held in a separate collection).
///
struct cell {
    channel_id channel0 = 0;
    channel_id channel1 = 0;
    scalar activation = 0.;
    scalar time = 0.;

    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
};

/// Declare all cell collection types
using cell_collection_types = collection_types<cell>;

TRACCC_HOST_DEVICE
inline bool operator<(const cell& lhs, const cell& rhs) {

    if (lhs.module_link != rhs.module_link) {
        return lhs.module_link < rhs.module_link;
    } else if (lhs.channel0 != rhs.channel0) {
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

    return ((lhs.module_link == rhs.module_link) &&
            (lhs.channel0 == rhs.channel0) && (lhs.channel1 == rhs.channel1) &&
            (std::abs(lhs.activation - rhs.activation) < float_epsilon) &&
            (std::abs(lhs.time - rhs.time) < float_epsilon));
}

}  // namespace traccc
