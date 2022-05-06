/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/seeding/detail/singlet.hpp"

namespace traccc {

/// Header: the number of doublets per spacepoint bin
struct doublet_per_bin {
    unsigned int n_doublets = 0;

    TRACCC_HOST_DEVICE
    unsigned int get_ref_num() const { return n_doublets; }

    TRACCC_HOST_DEVICE
    void zeros() { n_doublets = 0; }
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

// Declare all doublet collection/container types
TRACCC_DECLARE_COLLECTION_TYPES(doublet);
TRACCC_DECLARE_CONTAINER_TYPES(doublet, doublet_per_bin, doublet);

}  // namespace traccc
