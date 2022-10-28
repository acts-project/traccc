/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
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

/// @name Implementation for @c traccc::details::is_same_object<seed>
/// @{

is_same_object<seed>::is_same_object(
    const spacepoint_container_types::const_view& ref_spacepoints,
    const spacepoint_container_types::const_view& test_spacepoints,
    const seed& ref, scalar unc)
    : m_ref_spacepoints(ref.get_spacepoints(ref_spacepoints)),
      m_spacepoints(test_spacepoints),
      m_ref(ref),
      m_unc(unc) {}

bool is_same_object<seed>::operator()(const seed& obj) const {

    // Extract the spacepoints belonging to the tested seed.
    std::array<spacepoint, 3> test_spacepoints =
        obj.get_spacepoints(m_spacepoints);

    // Compare the two seeds.
    return (is_same_scalar(obj.weight, m_ref.get().weight, m_unc) &&
            is_same_scalar(obj.z_vertex, m_ref.get().z_vertex, m_unc) &&
            is_same_object<spacepoint>(m_ref_spacepoints[0],
                                       m_unc)(test_spacepoints[0]) &&
            is_same_object<spacepoint>(m_ref_spacepoints[1],
                                       m_unc)(test_spacepoints[1]) &&
            is_same_object<spacepoint>(m_ref_spacepoints[2],
                                       m_unc)(test_spacepoints[2]));
}

/// @}

/// @name Implementation for @c traccc::details::is_same_object<spacepoint>
/// @{

is_same_object<spacepoint>::is_same_object(const spacepoint& ref, scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<spacepoint>::operator()(const spacepoint& obj) const {

    return (is_same_scalar(obj.x(), m_ref.get().x(), m_unc) &&
            is_same_scalar(obj.y(), m_ref.get().y(), m_unc) &&
            is_same_scalar(obj.z(), m_ref.get().z(), m_unc) &&
            is_same_object<measurement>(m_ref.get().meas, m_unc)(obj.meas));
}

/// @}

/// @name Implementation for
///       @c traccc::details::is_same_object<bound_track_parameters>
/// @{

is_same_object<bound_track_parameters>::is_same_object(
    const bound_track_parameters& ref, scalar unc)
    : m_ref(ref), m_unc(unc) {}

bool is_same_object<bound_track_parameters>::operator()(
    const bound_track_parameters& obj) const {

    return ((obj.surface_link() == m_ref.get().surface_link()) &&
            is_same_scalar(obj.local()[0], m_ref.get().local()[0], m_unc) &&
            is_same_scalar(obj.local()[1], m_ref.get().local()[1], m_unc) &&
            is_same_angle(obj.phi(), m_ref.get().phi(), m_unc) &&
            is_same_scalar(obj.theta(), m_ref.get().theta(), m_unc) &&
            is_same_scalar(obj.time(), m_ref.get().time(), m_unc) &&
            is_same_scalar(obj.qop(), m_ref.get().qop(), m_unc));
}

/// @}

}  // namespace traccc::details
