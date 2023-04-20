/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/alt_measurement.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::measurement
template <>
class is_same_object<alt_measurement> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const alt_measurement& ref, scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::measurement
    bool operator()(const alt_measurement& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const alt_measurement> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<alt_measurement>

}  // namespace traccc::details
