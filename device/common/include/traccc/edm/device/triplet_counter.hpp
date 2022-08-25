/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/container.hpp"
#include "traccc/seeding/detail/doublet.hpp"

namespace traccc::device {

/// Header type for the "triplet counter container"
///
/// The header stores summary information about the number of triplets found in
/// a given geometric bin.
///
struct triplet_counter_header {

    /// The total number of middle-bottom spacepoint doublets in a given
    /// geometric bin.
    unsigned int m_nMidBot = 0;

    /// The total number of Triplets in a given geometric bin
    unsigned int m_nTriplets = 0;

};  // struct triplet_counter_header

/// Item type for the "triplet counter container"
///
/// It stores the number of triplets for one specific Mid Bottom Doublet.
///
struct triplet_counter {

    /// indices of two spacepoints of midbot doublet
    doublet m_midBotDoublet;

    /// The number of compatible triplets for a the midbot doublet
    unsigned int m_nTriplets = 0;

};  // struct triplet_counter

/// Declare all triplet counter collection types
using triplet_counter_collection_types = collection_types<triplet_counter>;
/// Declare all triplet counter container types
using triplet_counter_container_types =
    container_types<triplet_counter_header, triplet_counter>;

}  // namespace traccc::device
