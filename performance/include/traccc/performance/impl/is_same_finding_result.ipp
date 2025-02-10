/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_state.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::finding_result
template <>
class is_same_object<finding_result> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const finding_result& ref, scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::measurement
    bool operator()(const finding_result& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const finding_result> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<finding_result>

}  // namespace traccc::details
