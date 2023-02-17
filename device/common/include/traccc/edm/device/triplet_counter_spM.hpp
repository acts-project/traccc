/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/container.hpp"
#include "traccc/seeding/detail/singlet.hpp"

namespace traccc::device {

/// Header type for the "triplet counter container"
///
/// The header stores summary information about the number of triplets found in
/// a given geometric bin.
///
struct triplet_counter_spM_header {

    /// Index of the middle spacepoint.
    sp_location m_spM;

    /// The total number of Triplets in a given geometric bin
    unsigned int m_nTriplets = 0;

};  // struct triplet_counter_header

/// Item type for the "triplet counter container"
///
/// It stores the number of triplets for one specific Mid Bottom Doublet.
///
struct triplet_counter_spM {

    /// Index of the bottom spacepoint
    sp_location m_spB;

    /// The number of compatible triplets for a the midbot doublet
    unsigned int m_nTriplets = 0;

    /// The position in which these triplets will be added
    unsigned int posTriplets = 0;

};  // struct triplet_counter

/// Declare all triplet_counter_spM collection types
using triplet_counter_spM_collection_types =
    collection_types<triplet_counter_spM>;
/// Declare all triplet_counter_spM container types
using triplet_counter_spM_container_types =
    container_types<triplet_counter_spM_header, triplet_counter_spM>;

}  // namespace traccc::device