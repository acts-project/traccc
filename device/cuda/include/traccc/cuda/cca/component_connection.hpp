/*
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/utils/algorithm.hpp"

namespace traccc::cuda {
struct component_connection : algorithm<alt_measurement_collection_types::host(
                                  const cell_collection_types::host& data)> {
    output_type operator()(const cell_collection_types::host& data) const;
};
}  // namespace traccc::cuda
