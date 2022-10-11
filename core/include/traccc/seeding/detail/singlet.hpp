/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"

namespace traccc {

/// location of spacepoint in internal spacepoint container
struct sp_location {
    /// index of the bin of the spacepoint grid
    unsigned int bin_idx;
    /// index of the spacepoint in the bin
    unsigned int sp_idx;
};

inline TRACCC_HOST_DEVICE bool operator==(const sp_location& lhs,
                                          const sp_location& rhs) {
    return (lhs.bin_idx == rhs.bin_idx && lhs.sp_idx == rhs.sp_idx);
}

inline TRACCC_HOST_DEVICE bool operator!=(const sp_location& lhs,
                                          const sp_location& rhs) {
    return (lhs.bin_idx != rhs.bin_idx || lhs.sp_idx != rhs.sp_idx);
}

/// Declare all singlet collection types
using singlet_collection_types = collection_types<sp_location>;

/// Declare all singlet container types
using singlet_container_types = container_types<unsigned int, sp_location>;

}  // namespace traccc
