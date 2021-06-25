/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "utils/axis.hpp"
#include "utils/grid.hpp"

namespace traccc {

// define spacepoint_grid
using spacepoint_grid = grid<traccc::axis<AxisBoundaryType::Closed>,
                             traccc::axis<AxisBoundaryType::Bound> >;

}  // namespace traccc
