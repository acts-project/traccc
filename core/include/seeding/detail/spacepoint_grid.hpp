/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// ActsCore
#include "Acts/Utilities/detail/Axis.hpp"
#include "Acts/Utilities/detail/Grid.hpp"

namespace traccc {

// define spacepoint_grid
using spacepoint_grid = Acts::detail::Grid<
    int,
    Acts::detail::Axis<Acts::detail::AxisType::Equidistant,
                       Acts::detail::AxisBoundaryType::Closed>,
    Acts::detail::Axis<Acts::detail::AxisType::Equidistant,
                       Acts::detail::AxisBoundaryType::Bound> >;

}  // namespace traccc
