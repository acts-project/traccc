/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_state.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::fitting_result
template <>
class is_same_object<fitting_result<transform3>> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const fitting_result<transform3>& ref,
                   scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::measurement
    bool operator()(const fitting_result<transform3>& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const fitting_result<transform3>> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<fitting_result<transform3>>

}  // namespace traccc::details
