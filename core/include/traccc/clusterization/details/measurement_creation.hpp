/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

namespace traccc::details {

/// Function used for retrieving the cell signal based on the module id
TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(
    scalar signal_in, const silicon_detector_description::const_device&);

/// Function for pixel segmentation
TRACCC_HOST_DEVICE
inline vector2 position_from_cell(
    const cell& cell,
    const silicon_detector_description::const_device& det_descr);

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[in] cluster   The vector of cells describing the identified cluster
/// @param[in] det_descr The detector description
/// @param[out] mean     The mean position of the cluster/measurement
/// @param[out] var      The variation on the mean position of the
///                      cluster/measurement
/// @param[out] totalWeight The total weight of the cluster/measurement
///
TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const cell_collection_types::const_device& cluster,
    const silicon_detector_description::const_device& det_descr, point2& mean,
    point2& var, scalar& totalWeight);

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[out] measurements Measurement collection where the measurement is to
///                          be filled
/// @param[in] index     Index of the measurement object to fill
/// @param[in] cluster   Cell vector describing the measurement's cluster
/// @param[in] det_descr Detector description
///
TRACCC_HOST_DEVICE inline void fill_measurement(
    measurement_collection_types::device& measurements,
    measurement_collection_types::device::size_type index,
    const cell_collection_types::const_device& cluster,
    const silicon_detector_description::const_device& det_descr);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/clusterization/impl/measurement_creation.ipp"
