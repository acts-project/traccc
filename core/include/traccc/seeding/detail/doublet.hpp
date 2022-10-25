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

    // Position of the mid top doublets for this which share this spM
    unsigned int m_mt_start_idx = 0;
    unsigned int m_mt_end_idx = 0;
};

inline TRACCC_HOST_DEVICE bool operator==(const doublet& lhs,
                                          const doublet& rhs) {
    return (lhs.sp1.bin_idx == rhs.sp1.bin_idx &&
            lhs.sp1.sp_idx == rhs.sp1.sp_idx &&
            lhs.sp2.bin_idx == rhs.sp2.bin_idx &&
            lhs.sp2.sp_idx == rhs.sp2.sp_idx);
}

/// Declare all doublet collection types
using doublet_collection_types = collection_types<doublet>;

/// Declare all doublet container types
using doublet_container_types = container_types<doublet_per_bin, doublet>;

}  // namespace traccc
