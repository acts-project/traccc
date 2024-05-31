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

namespace traccc::details {

/// Function used for retrieving the cell signal based on the module id
TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(scalar signal_in, const cell_module& mod);

/// Function for pixel segmentation
TRACCC_HOST_DEVICE
inline vector2 position_from_cell(const cell& cell, const cell_module& mod);

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[in] cluster The vector of cells describing the identified cluster
/// @param[in] mod     The cell module
/// @param[out] mean   The mean position of the cluster/measurement
/// @param[out] var    The variation on the mean position of the
///                    cluster/measurement
/// @param[out] totalWeight The total weight of the cluster/measurement
///
TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const cell_collection_types::const_device& cluster, const cell_module& mod,
    point2& mean, point2& var, scalar& totalWeight);

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[out] measurements is the measurement collection where the measurement
///                          object will be filled
/// @param[in] measurement_index is the index of the measurement object to fill
/// @param[in] cluster is the input cell vector
/// @param[in] mod  is the cell module where the cluster belongs to
/// @param[in] mod_link is the module index
///
TRACCC_HOST_DEVICE inline void fill_measurement(
    measurement_collection_types::device& measurements,
    std::size_t measurement_index,
    const cell_collection_types::const_device& cluster, const cell_module& mod,
    const unsigned int mod_link);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/clusterization/impl/measurement_creation.ipp"
