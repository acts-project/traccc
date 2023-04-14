/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/fill_prefix_sum.hpp"
#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function for creating 3D spacepoints out of 2D measurements
///
/// @param[in] globalIndex          The index for the current thread
/// @param[in] measurements_view    Collection of measurements
/// @param[in] modules_view         Collection of modules (which the
/// measurements link to)
/// @param[in] measurement_count    Number of measurements
/// @param[out] spacepoints_view    Collection of spacepoints
///
TRACCC_HOST_DEVICE
inline void form_spacepoints(
    const std::size_t globalIndex,
    alt_measurement_collection_types::const_view measurements_view,
    cell_module_collection_types::const_view modules_view,
    const unsigned int measurement_count,
    spacepoint_collection_types::view spacepoints_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/form_spacepoints.ipp"