/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

namespace traccc {

/// Alternative cell structure which contains both the cell and a link to its
/// module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the cell
/// container types
struct alt_cell {
    cell c;
    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
};

/// Declare all cell collection types
using alt_cell_collection_types = collection_types<alt_cell>;

TRACCC_HOST_DEVICE
inline bool operator<(const alt_cell& lhs, const alt_cell& rhs) {

    if (lhs.module_link != rhs.module_link) {
        return lhs.module_link < rhs.module_link;
    } else if (lhs.c.channel0 != rhs.c.channel0) {
        return (lhs.c.channel0 < rhs.c.channel0);
    } else if (lhs.c.channel1 != rhs.c.channel1) {
        return (lhs.c.channel1 < rhs.c.channel1);
    } else {
        return lhs.c.activation < rhs.c.activation;
    }
}

/// Equality operator for cells
TRACCC_HOST_DEVICE
inline bool operator==(const alt_cell& lhs, const alt_cell& rhs) {

    return ((lhs.module_link == rhs.module_link) &&
            (lhs.c.channel0 == rhs.c.channel0) &&
            (lhs.c.channel1 == rhs.c.channel1) &&
            (std::abs(lhs.c.activation - rhs.c.activation) < float_epsilon) &&
            (std::abs(lhs.c.time - rhs.c.time) < float_epsilon));
}

}  // namespace traccc
