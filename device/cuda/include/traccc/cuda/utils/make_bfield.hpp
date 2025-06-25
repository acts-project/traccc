/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/bfield.hpp"

namespace traccc::cuda {

/// Create a magnetic field usable on the active CUDA device
///
/// @param field The magnetic field to be copied
/// @return A copy of the magnetic field that can be used on the active CUDA
///         device
///
bfield make_bfield(const bfield& field);

}  // namespace traccc::cuda
