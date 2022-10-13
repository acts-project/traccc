/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

/// Function used for creating the 2D measurement objects out of the clusters in
/// each module
///
/// The output is a measurement container with 1 measurement corresponding to 1
/// cluster
///
/// @param[in] globalIndex          The index for the current thread
/// @param[in] clusters_view        Container storing the cells for every
/// cluster
/// @param[in] cells_view           The cells for each module
/// @param[out] measurements_view   Container storing the created measurements
/// for each module
///
TRACCC_HOST_DEVICE
inline void create_measurements(
    std::size_t globalIndex, cluster_container_types::const_view clusters_view,
    const cell_container_types::const_view& cells_view,
    measurement_container_types::view measurements_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/clusterization/device/impl/create_measurements.ipp"