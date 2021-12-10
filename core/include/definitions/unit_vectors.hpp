/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/primitives.hpp>
#include <definitions/qualifiers.hpp>

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

}  // namespace traccc
