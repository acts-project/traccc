/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/details/is_same_object.hpp"

#include "traccc/performance/details/is_same_angle.hpp"
#include "traccc/performance/details/is_same_scalar.hpp"

namespace traccc::details {

/// @name Implementation for @c traccc::details::is_same_object<measurement>
/// @{

is_same_object<measurement>::is_same_object(const measurement& ref, scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<measurement>::operator()(const measurement& obj) const {

    return (is_same_scalar(obj.local[0], m_ref.get().local[0], m_unc) &&
            is_same_scalar(obj.local[1], m_ref.get().local[1], m_unc) &&
            is_same_scalar(obj.variance[0], m_ref.get().variance[0], m_unc) &&
            is_same_scalar(obj.variance[1], m_ref.get().variance[1], m_unc));
}

/// @}

/// @name Implementation for
///       @c traccc::details::is_same_object<bound_track_parameters>
/// @{
is_same_object<bound_track_parameters<>>::is_same_object(
    const bound_track_parameters<>& ref, scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<bound_track_parameters<>>::operator()(
    const bound_track_parameters<>& obj) const {

    return ((obj.surface_link() == m_ref.get().surface_link()) &&
            is_same_scalar(obj.bound_local()[0], m_ref.get().bound_local()[0],
                           m_unc) &&
            is_same_scalar(obj.bound_local()[1], m_ref.get().bound_local()[1],
                           m_unc) &&
            is_same_angle(obj.phi(), m_ref.get().phi(), m_unc) &&
            is_same_scalar(obj.theta(), m_ref.get().theta(), m_unc) &&
            is_same_scalar(obj.time(), m_ref.get().time(), m_unc) &&
            is_same_scalar(obj.qop(), m_ref.get().qop(), m_unc));
}

/// @}

/// @name Implementation for
///       @c
///       traccc::details::is_same_object<track_candidate_collection_types::host>
/// @{

is_same_object<track_candidate_collection_types::host>::is_same_object(
    const track_candidate_collection_types::host& ref, scalar)
    : m_ref(ref) {}

bool is_same_object<track_candidate_collection_types::host>::operator()(
    const track_candidate_collection_types::host& obj) const {

    const track_candidate_collection_types::host::size_type n_cands =
        m_ref.get().size();
    for (track_candidate_collection_types::host::size_type i = 0; i < n_cands;
         i++) {

        const bool is_same = m_ref.get()[i] == obj[i];

        if (!is_same) {
            return false;
        }
    }

    return true;
}

/// @}

/// @name Implementation for
///       @c traccc::details::is_same_object<fitting_result>
/// @{

is_same_object<fitting_result<traccc::default_algebra>>::is_same_object(
    const fitting_result<traccc::default_algebra>& ref, scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<fitting_result<traccc::default_algebra>>::operator()(
    const fitting_result<traccc::default_algebra>& obj) const {

    return (is_same_object<bound_track_parameters<>>(m_ref.get().fit_params,
                                                     m_unc)(obj.fit_params) &&
            is_same_scalar(obj.trk_quality.ndf, m_ref.get().trk_quality.ndf,
                           m_unc));
}

/// @}

/// @name Implementation for
///       @c traccc::details::is_same_object<finding_result>
/// @{

is_same_object<finding_result>::is_same_object(const finding_result& ref,
                                               scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<finding_result>::operator()(
    const finding_result& obj) const {

    return (is_same_object<bound_track_parameters<>>(m_ref.get().seed_params,
                                                     m_unc)(obj.seed_params) &&
            is_same_scalar(obj.trk_quality.ndf, m_ref.get().trk_quality.ndf,
                           m_unc));
}

/// @}

}  // namespace traccc::details
