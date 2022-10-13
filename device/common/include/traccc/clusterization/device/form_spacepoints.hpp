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
/// @param[in] globalIndex                  The index for the current thread
/// @param[in] measurements_view            Container storing the created
/// measurements for each module
/// @param[in] measurements_prefix_sum_view Prefix sum for iterating over all
/// measurements
/// @param[out] spacepoints_view            Container storing #D spacepoints for
/// each module
///
TRACCC_HOST_DEVICE
inline void form_spacepoints(
    std::size_t globalIndex,
    measurement_container_types::const_view measurements_view,
    vecmem::data::vector_view<const device::prefix_sum_element_t>
        measurements_prefix_sum_view,
    spacepoint_container_types::view spacepoints_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/form_spacepoints.ipp"