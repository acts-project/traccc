/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/seeding/detail/singlet.hpp"

namespace traccc {

/// Header: the number of triplets per spacepoint bin
struct triplet_per_bin {
    unsigned int n_triplets = 0;

    TRACCC_HOST_DEVICE
    unsigned int get_ref_num() const { return n_triplets; }

    TRACCC_HOST_DEVICE
    void zeros() { n_triplets = 0; }
};

/// Item: triplets of middle-bottom-top
struct triplet {
    // middle spacepoint location in internal spacepoint container
    sp_location sp1;
    // bottom spacepoint location in internal spacepoint container
    sp_location sp2;
    // top spacepoint location in internal spacepoint container
    sp_location sp3;
    // curvtaure of circle estimated from triplet
    scalar curvature;
    // weight of triplet
    scalar weight;
    // z origin of triplet
    scalar z_vertex;
};

/// Equality operator for triplets
inline TRACCC_HOST_DEVICE bool operator==(const triplet& lhs,
                                          const triplet& rhs) {
    return ((lhs.sp1 == rhs.sp1) && (lhs.sp2 == rhs.sp2) &&
            (lhs.sp3 == rhs.sp3));
}

/// Comparison / ordering operator for triplets
inline TRACCC_HOST_DEVICE bool operator<(const triplet& lhs,
                                         const triplet& rhs) {
    return lhs.weight < rhs.weight;
}

// Declare all triplet collection/container types
TRACCC_DECLARE_COLLECTION_TYPES(triplet);
TRACCC_DECLARE_CONTAINER_TYPES(triplet, triplet_per_bin, triplet);

}  // namespace traccc
