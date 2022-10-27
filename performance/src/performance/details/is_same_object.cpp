/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/performance/details/is_same_object.hpp"

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

}  // namespace traccc::details
