/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/magnetic_field.hpp"
#include "traccc/utils/bfield.hpp"

namespace traccc::details {

/// Create a magnetic field object based on the provided options
///
/// @param opts The command line options for the magnetic field
/// @return A magnetic field object configured according to the options
///
bfield make_magnetic_field(const opts::magnetic_field& opts);

}  // namespace traccc::details
