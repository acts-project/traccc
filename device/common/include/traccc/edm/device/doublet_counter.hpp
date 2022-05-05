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

    /// Temporary compatibility function
    TRACCC_HOST_DEVICE
    unsigned int get_ref_num() const { return m_nSpM; }

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

/// Convenience declaration for the doublet_counter collection type to use in
/// host code
using host_doublet_counter_collection = vecmem::vector<doublet_counter>;

/// Convenience declaration for the doublet_counter collection type to use in
/// device code (non-const)
using device_doublet_counter_collection =
    vecmem::device_vector<doublet_counter>;

/// Convenience declaration for the doublet_counter collection type to use in
/// device code (const)
using device_doublet_counter_const_collection =
    vecmem::device_vector<const doublet_counter>;

/// Convenience declaration for the doublet_counter container type to use in
/// host code
using host_doublet_counter_container =
    host_container<doublet_counter_header, doublet_counter>;

/// Convenience declaration for the doublet_counter container type to use in
/// device code (non-const)
using device_doublet_counter_container =
    device_container<doublet_counter_header, doublet_counter>;

/// Convenience declaration for the doublet_counter container type to use in
/// device code (const)
using device_doublet_counter_const_container =
    device_container<const doublet_counter_header, const doublet_counter>;

/// Convenience declaration for the doublet_counter container data type to use
/// in host code (non-const)
using doublet_counter_container_data =
    container_data<doublet_counter_header, doublet_counter>;

/// Convenience declaration for the doublet_counter container data type to use
/// in host code (const)
using doublet_counter_container_const_data =
    container_data<const doublet_counter_header, const doublet_counter>;

/// Convenience declaration for the doublet_counter container buffer type to use
/// in host code
using doublet_counter_container_buffer =
    container_buffer<doublet_counter_header, doublet_counter>;

/// Convenience declaration for the doublet_counter container view type to use
/// in host code (non-const)
using doublet_counter_container_view =
    container_view<doublet_counter_header, doublet_counter>;

/// Convenience declaration for the doublet_counter container view type to use
/// in host code (const)
using doublet_counter_container_const_view =
    container_view<const doublet_counter_header, const doublet_counter>;

}  // namespace traccc::device
