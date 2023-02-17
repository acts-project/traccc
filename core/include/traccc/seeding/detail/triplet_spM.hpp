/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/seeding/detail/singlet.hpp"

namespace traccc {

/// Header: the number of triplets per spacepoint middle
struct triplets_per_spM {
    sp_location m_spM;

    unsigned int n_triplets = 0;
};

/// Item: triplets of middle-bottom-top
struct triplet_spM {
    // bottom spacepoint location in internal spacepoint container
    sp_location spB;
    // top spacepoint location in internal spacepoint container
    sp_location spT;
    // curvtaure of circle estimated from triplet
    scalar curvature;
    // weight of triplet
    scalar weight;
    // z origin of triplet
    scalar z_vertex;

    /// Position of the triplets which share this mb doublet
    unsigned int triplets_mb_begin = 0;
    unsigned int triplets_mb_end = 0;
};

inline TRACCC_HOST_DEVICE bool operator==(const triplet_spM& lhs,
                                          const triplet_spM& rhs) {
    return (lhs.spB.bin_idx == rhs.spB.bin_idx &&
            lhs.spB.sp_idx == rhs.spB.sp_idx &&
            lhs.spT.bin_idx == rhs.spT.bin_idx &&
            lhs.spT.sp_idx == rhs.spT.sp_idx);
}

inline TRACCC_HOST_DEVICE bool operator!=(const triplet_spM& lhs,
                                          const triplet_spM& rhs) {
    return !(lhs == rhs);
}

inline TRACCC_HOST_DEVICE bool operator<(const triplet_spM& lhs,
                                         const triplet_spM& rhs) {
    return lhs.weight < rhs.weight;
}

/// Declare all triplet spM collection types
using triplet_spM_collection_types = collection_types<triplet_spM>;

/// Declare all triplet spM container types
using triplet_spM_container_types =
    container_types<triplets_per_spM, triplet_spM>;

}  // namespace traccc