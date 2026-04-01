/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/math.hpp"

namespace traccc::edm {

template <typename BASE>
template <detray::concepts::algebra ALGEBRA_TYPE>
TRACCC_HOST_DEVICE detray::dpoint2D<ALGEBRA_TYPE>
measurement<BASE>::local_position_in() const {

    detray::dpoint2D<ALGEBRA_TYPE> result;
    getter::element(result, 0) =
        static_cast<typename ALGEBRA_TYPE::value_type>(local_position()[0]);
    getter::element(result, 1) =
        static_cast<typename ALGEBRA_TYPE::value_type>(local_position()[1]);
    return result;
}

template <typename BASE>
template <detray::concepts::algebra ALGEBRA_TYPE>
TRACCC_HOST_DEVICE void measurement<BASE>::set_local_position_in(
    const detray::dpoint2D<ALGEBRA_TYPE>& pos) {

    local_position()[0] = static_cast<float>(getter::element(pos, 0));
    local_position()[1] = static_cast<float>(getter::element(pos, 1));
}

template <typename BASE>
template <detray::concepts::algebra ALGEBRA_TYPE>
TRACCC_HOST_DEVICE detray::dpoint2D<ALGEBRA_TYPE>
measurement<BASE>::local_variance_in() const {

    detray::dpoint2D<ALGEBRA_TYPE> result;
    getter::element(result, 0) =
        static_cast<typename ALGEBRA_TYPE::value_type>(local_variance()[0]);
    getter::element(result, 1) =
        static_cast<typename ALGEBRA_TYPE::value_type>(local_variance()[1]);
    return result;
}

template <typename BASE>
template <detray::concepts::algebra ALGEBRA_TYPE>
TRACCC_HOST_DEVICE void measurement<BASE>::set_local_variance_in(
    const detray::dpoint2D<ALGEBRA_TYPE>& var) {

    local_variance()[0] = static_cast<float>(getter::element(var, 0));
    local_variance()[1] = static_cast<float>(getter::element(var, 1));
}

template <typename BASE>
template <std::integral TYPE>
TRACCC_HOST_DEVICE void measurement<BASE>::set_subspace(
    const std::array<TYPE, 2u>& subs) {

    subspace()[0] = static_cast<unsigned int>(subs[0]);
    subspace()[1] = static_cast<unsigned int>(subs[1]);
}

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE bool measurement<BASE>::operator==(
    const measurement<T>& other) const {

    return ((surface_link() == other.surface_link()) &&
            (math::abs(local_position()[0] - other.local_position()[0]) <
             float_epsilon) &&
            (math::abs(local_position()[1] - other.local_position()[1]) <
             float_epsilon) &&
            (math::abs(local_variance()[0] - other.local_variance()[0]) <
             float_epsilon) &&
            (math::abs(local_variance()[1] - other.local_variance()[1]) <
             float_epsilon));
}

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE std::partial_ordering measurement<BASE>::operator<=>(
    const measurement<T>& other) const {

    if (surface_link() != other.surface_link()) {
        return (surface_link() <=> other.surface_link());
    } else if (local_position()[0] != other.local_position()[0]) {
        return (local_position()[0] <=> other.local_position()[0]);
    } else if (local_position()[1] != other.local_position()[1]) {
        return (local_position()[1] <=> other.local_position()[1]);
    } else if (local_variance()[0] != other.local_variance()[0]) {
        return (local_variance()[0] <=> other.local_variance()[0]);
    } else {
        return (local_variance()[1] <=> other.local_variance()[1]);
    }
}

}  // namespace traccc::edm
