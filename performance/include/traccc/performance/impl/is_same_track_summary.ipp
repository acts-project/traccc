/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_candidate.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::track_summary
template <>
class is_same_object<track_summary> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const track_summary& ref, scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::measurement
    bool operator()(const track_summary& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const track_summary> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<track_summary>

}  // namespace traccc::details
