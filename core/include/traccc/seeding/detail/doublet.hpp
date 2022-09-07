/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/seeding/detail/singlet.hpp"

namespace traccc {

/// Header: the number of doublets per spacepoint bin
struct doublet_per_bin {
    unsigned int n_doublets = 0;
};

/// Item: doublet of middle-bottom or middle-top
struct doublet {
    // midle spacepoint location in internal spacepoint container
    sp_location sp1;
    // bottom (or top) spacepoint location in internal spacepoint container
    sp_location sp2;
};

inline TRACCC_HOST_DEVICE bool operator==(const doublet& lhs,
                                          const doublet& rhs) {
    return (lhs.sp1.bin_idx == rhs.sp1.bin_idx &&
            lhs.sp1.sp_idx == rhs.sp1.sp_idx &&
            lhs.sp2.bin_idx == rhs.sp2.bin_idx &&
            lhs.sp2.sp_idx == rhs.sp2.sp_idx);
}

/// Container of doublet belonging to one detector module
template <template <typename> class vector_t>
using doublet_collection = vector_t<doublet>;

/// Convenience declaration for the doublet collection type to use in host code
using host_doublet_collection = doublet_collection<vecmem::vector>;

/// Convenience declaration for the doublet collection type to use in device
/// code
using device_doublet_collection = doublet_collection<vecmem::device_vector>;

/// Convenience declaration for the doublet container type to use in host code
using host_doublet_container = host_container<doublet_per_bin, doublet>;

/// Convenience declaration for the doublet container type to use in device code
using device_doublet_container = device_container<doublet_per_bin, doublet>;

/// Convenience declaration for the doublet container data type to use in host
/// code
using doublet_container_data = container_data<doublet_per_bin, doublet>;

/// Convenience declaration for the doublet container buffer type to use in host
/// code
using doublet_container_buffer = container_buffer<doublet_per_bin, doublet>;

/// Convenience declaration for the doublet container view type to use in host
/// code
using doublet_container_view = container_view<doublet_per_bin, doublet>;

}  // namespace traccc
