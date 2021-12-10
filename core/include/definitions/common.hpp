/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <definitions/primitives.hpp>

#include "Acts/Definitions/Units.hpp"

namespace traccc {

// epsilon for float variables
constexpr scalar float_epsilon = 1e-5;

// pion mass for track parameter estimation
constexpr scalar PION_MASS_MEV = 139.57018 * Acts::UnitConstants::MeV;

}  // namespace traccc
