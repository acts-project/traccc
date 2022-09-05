/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/seeding/detail/singlet.hpp"

namespace traccc::device {

/// Header type for the "doublet container"
///
/// The header stores summary information about the number of doublets found in
/// a given geometric bin.
///
struct doublet_counter_header {

    /// The total number of middle spacepoints in a given geometric bin for
    /// which a compatible bottom- or top-doublet was found.
    unsigned int m_nSpM = 0;

    /// The total number of middle-bottom spacepoint doublets in a given
    /// geometric bin.
    unsigned int m_nMidBot = 0;

    /// The total number of middle-top spacepoint doublets in a given
    /// geometric bin.
    unsigned int m_nMidTop = 0;

};  // struct doublet_counter_header

/// Item type for the "doublet container"
///
/// It stores the number of doublets for one specific middle spacepoint.
///
struct doublet_counter {

    /// Index of the middle spacepoint.
    sp_location m_spM;

    /// The number of compatible middle-bottom doublets for the middle
    /// spacepoint.
    unsigned int m_nMidBot = 0;

    /// The number of compatible middle-top doublets for a the middle
    /// spacepoint.
    unsigned int m_nMidTop = 0;

};  // struct doublet_counter

/// Declare all doublet counter collection types
using doublet_counter_collection_types = collection_types<doublet_counter>;
/// Declare all doublet counter container types
using doublet_counter_container_types =
    container_types<doublet_counter_header, doublet_counter>;

}  // namespace traccc::device
