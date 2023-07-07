/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/track_candidate.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for
/// @c traccc::track_candidate_collection_types::host
template <>
class is_same_object<track_candidate_collection_types::host> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const track_candidate_collection_types::host& ref,
                   scalar unc = float_epsilon);

    /// Specialised implementation for @c traccc::bound_track_parameters
    bool operator()(const track_candidate_collection_types::host& obj) const;

    private:
    /// The reference object
    std::reference_wrapper<const track_candidate_collection_types::host> m_ref;
    /// The uncertainty
    scalar m_unc;

};  // class is_same_object<track_candidate_collection_types::host>

}  // namespace traccc::details
