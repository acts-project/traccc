/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"

// System include(s).
#include <array>

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::seed
template <>
class is_same_object<seed> {

    public:
    /// Constructor with all necessary arguments
    is_same_object(
        const spacepoint_container_types::const_view& ref_spacepoints,
        const spacepoint_container_types::const_view& test_spacepoints,
        const seed& ref, scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::seed
    bool operator()(const seed& obj) const;

    private:
    /// Spacepoints for the reference object
    const std::array<spacepoint, 3> m_ref_spacepoints;
    /// Spacepoint container for the test seeds
    const spacepoint_container_types::const_view m_spacepoints;

    /// The reference object
    std::reference_wrapper<const seed> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<seed>

}  // namespace traccc::details
