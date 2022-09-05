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

/// Container of singlet belonging to one detector module
template <template <typename> class vector_t>
using singlet_collection = vector_t<sp_location>;

/// Convenience declaration for the singlet collection type to use in host code
using host_singlet_collection = singlet_collection<vecmem::vector>;

/// Convenience declaration for the singlet collection type to use in device
/// code
using device_singlet_collection = singlet_collection<vecmem::device_vector>;

/// Convenience declaration for the singlet container type to use in host code
using host_singlet_container = host_container<unsigned int, sp_location>;

/// Convenience declaration for the singlet container type to use in device code
using device_singlet_container = device_container<unsigned int, sp_location>;

/// Convenience declaration for the singlet container data type to use in host
/// code
using singlet_container_data = container_data<unsigned int, sp_location>;

/// Convenience declaration for the singlet container buffer type to use in host
/// code
using singlet_container_buffer = container_buffer<unsigned int, sp_location>;

/// Convenience declaration for the singlet container view type to use in host
/// code
using singlet_container_view = container_view<unsigned int, sp_location>;
}  // namespace traccc
