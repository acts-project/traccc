/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/seeding/detail/singlet.hpp"

namespace traccc {

/// Header: the number of doublets per spacepoint bin
struct doublets_per_spM {

    /// Index of the middle spacepoint.
    sp_location spM;

    /// The number of compatible middle-bottom/top doublets for the middle
    /// spacepoint.
    unsigned int n_doublets = 0;
};

/// Item: doublet of middle-bottom or middle-top
struct doublet_spM {
    // bottom (or top) spacepoint location in internal spacepoint container
    sp_location sp2;
};

inline TRACCC_HOST_DEVICE bool operator==(const doublet_spM& lhs,
                                          const doublet_spM& rhs) {
    return (lhs.sp2.bin_idx == rhs.sp2.bin_idx &&
            lhs.sp2.sp_idx == rhs.sp2.sp_idx);
}

/// Declare all doublet collection types
using doublet_spM_collection_types = collection_types<doublet_spM>;

/// Declare all doublet container types
using doublet_spM_container_types =
    container_types<doublets_per_spM, doublet_spM>;

}  // namespace traccc