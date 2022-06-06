/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::kokkos {
struct component_connection : algorithm<measurement_container_types::host(
                                  const cell_container_types::host& cells)> {
    output_type operator()(const cell_container_types::host& cells) const;
};
}  // namespace traccc::kokkos
