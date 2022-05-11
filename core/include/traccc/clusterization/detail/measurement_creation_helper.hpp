/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"

namespace traccc::detail {

/// Function used for retrieving the cell signal based on the module id
TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cluster_id& /*cl_id*/) {
    return signal_in;
}

/// Function for pixel segmentation
TRACCC_HOST_DEVICE
inline vector2 position_from_cell(const cell& c, const cluster_id& cl_id) {
    // Retrieve the specific values based on module idx
    return {cl_id.pixel.min_center_x + c.channel0 * cl_id.pixel.pitch_x,
            cl_id.pixel.min_center_y + c.channel1 * cl_id.pixel.pitch_y};
}

TRACCC_HOST_DEVICE
inline vector2 get_pitch(const cluster_id& cl_id) {
    // return the values based on the module idx
    return {cl_id.pixel.pitch_x, cl_id.pixel.pitch_y};
}

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[in] cluster The vector of cells describing the identified cluster
/// @param[in] cl_id   The cluster identifier
/// @param[out] mean   The mean position of the cluster/measurement
/// @param[out] var    The variation on the mean position of the
///                    cluster/measurement
/// @param[out] totalWeight The total weight of the cluster/measurement
///
TRACCC_HOST_DEVICE inline void calc_cluster_properties(
    const cell_collection_types::const_device& cluster, const cluster_id& cl_id,
    point2& mean, point2& var, scalar& totalWeight) {

    // Loop over the cells of the cluster.
    for (const cell& cell : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight = signal_cell_modelling(cell.activation, cl_id);

        // Only consider cells over a minimum threshold.
        if (weight > cl_id.threshold) {

            // Update all output properties with this cell.
            totalWeight += cell.activation;
            const point2 cell_position = position_from_cell(cell, cl_id);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (std::size_t i = 0; i < 2; ++i) {
                var[i] =
                    var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
            }
        }
    }
}

}  // namespace traccc::detail
