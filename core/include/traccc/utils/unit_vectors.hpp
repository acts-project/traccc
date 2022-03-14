/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

TRACCC_HOST_DEVICE inline vector3 make_direction_unit_from_phi_theta(
    scalar phi, scalar theta) {
    const auto cosTheta = std::cos(theta);
    const auto sinTheta = std::sin(theta);
    return {
        std::cos(phi) * sinTheta,
        std::sin(phi) * sinTheta,
        cosTheta,
    };
}

TRACCC_HOST_DEVICE inline scalar phi(const vector3& v) {
    return std::atan2(v[1], v[0]);
}

TRACCC_HOST_DEVICE inline scalar eta(const vector3& v) {
    return std::atanh(v[2] / getter::norm(v));
}

TRACCC_HOST_DEVICE inline scalar theta(const vector3& v) {
    return std::atan2(std::sqrt(v[0] * v[0] + v[1] * v[1]), v[2]);
}

}  // namespace traccc
