/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"

// System include(s).
#include <cstddef>
#include <variant>

namespace traccc {

/// Declare all cluster container types
using cluster_container_types = container_types<std::monostate, cell>;

}  // namespace traccc
