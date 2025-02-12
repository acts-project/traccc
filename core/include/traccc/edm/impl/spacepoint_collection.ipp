/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/math.hpp"

namespace traccc::edm {

template <typename BASE>
TRACCC_HOST_DEVICE auto spacepoint<BASE>::radius() const {

    return math::sqrt(x() * x() + y() * y());
}

template <typename BASE>
TRACCC_HOST_DEVICE auto spacepoint<BASE>::phi() const {

    return math::atan2(y(), x());
}

template <typename BASE>
TRACCC_HOST_DEVICE auto spacepoint<BASE>::global() const {

    return point3{x(), y(), z()};
}

template <typename BASE>
TRACCC_HOST_DEVICE spacepoint<BASE>& spacepoint<BASE>::operator=(
    const spacepoint& other) {

    measurement_index() = other.measurement_index();
    x() = other.x();
    y() = other.y();
    z() = other.z();
    z_variance() = other.z_variance();
    radius_variance() = other.radius_variance();
    return *this;
}

template <typename BASE>
template <typename T, std::enable_if_t<!std::is_same_v<BASE, T>, bool> >
TRACCC_HOST_DEVICE spacepoint<BASE>& spacepoint<BASE>::operator=(
    const spacepoint<T>& other) {

    measurement_index() = other.measurement_index();
    x() = other.x();
    y() = other.y();
    z() = other.z();
    z_variance() = other.z_variance();
    radius_variance() = other.radius_variance();
    return *this;
}

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE bool spacepoint<BASE>::operator==(
    const spacepoint<T>& other) const {

    return ((measurement_index() == other.measurement_index()) &&
            (math::fabs(x() - other.x()) < 1e-6f) &&
            (math::fabs(y() - other.y()) < 1e-6f) &&
            (math::fabs(z() - other.z()) < 1e-6f) &&
            (z_variance() == other.z_variance()) &&
            (radius_variance() == other.radius_variance()));
}

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE std::weak_ordering spacepoint<BASE>::operator<=>(
    const spacepoint<T>& other) const {

    return (radius() <=> other.radius());
}

}  // namespace traccc::edm
