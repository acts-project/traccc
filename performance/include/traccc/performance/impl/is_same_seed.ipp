/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/seed.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::seed
template <>
class is_same_object<seed> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const seed& ref, scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::seed
    bool operator()(const seed& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const seed> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<seed>

}  // namespace traccc::details
