/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cluster.hpp"

namespace traccc {

TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cluster_id& /*cl_id*/) {
    // Retrieve the signal based on the cl_id.module_idx
    return signal_in;
}

/// Function for pixel segmentation
TRACCC_HOST_DEVICE
inline vector2 position_from_cell(cell c, cluster_id cl_id) {
    // Retrieve the specific values based on module idx
    return {cl_id.pixel.min_center_x + c.channel0 * cl_id.pixel.pitch_x,
            cl_id.pixel.min_center_y + c.channel1 * cl_id.pixel.pitch_y};
}

TRACCC_HOST_DEVICE
inline vector2 get_pitch(cluster_id cl_id) {
    // return the values based on the module idx
    return {cl_id.pixel.pitch_x, cl_id.pixel.pitch_y};
}

TRACCC_HOST_DEVICE
inline void calc_cluster_properties(const cell& c, const cluster_id& cl_id,
                                    point2& mean, point2& var,
                                    scalar& totalWeight) {
    scalar weight = signal_cell_modelling(c.activation, cl_id);
    if (weight > cl_id.threshold) {
        totalWeight += c.activation;
        const point2 cell_position = position_from_cell(c, cl_id);
        const point2 prev = mean;
        const point2 diff = cell_position - prev;

        mean = prev + (weight / totalWeight) * diff;
        for (std::size_t i = 0; i < 2; ++i) {
            var[i] = var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
        }
    }
}

}  // namespace traccc
