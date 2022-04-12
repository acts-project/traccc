/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/geometry/pixel_data.hpp"

// System include(s).
#include <functional>
#include <vector>

namespace traccc {

struct cluster_id {

    event_id event = 0;
    std::size_t module_idx = 0;
    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0.;

    pixel_data pixel;
    bool is_default = true;
};

/// Convenience declaration for the cluster container type to use in host code
using host_cluster_container = host_container<cluster_id, cell>;

/// Convenience declaration for the cluster container type to use in device code
using device_cluster_container = device_container<cluster_id, cell>;

/// Convenience declaration for the cluster container type to use in device code
/// (const)
using device_cluster_const_container =
    device_container<const cluster_id, const cell>;

/// Convenience declaration for the cluster container data type to use in host
/// code
using cluster_container_data = container_data<cluster_id, cell>;

/// Convenience declaration for the cluster container data type to use in host
/// code (const)
using cluster_container_const_data =
    container_data<const cluster_id, const cell>;

/// Convenience declaration for the cluster container buffer type to use in host
/// code
using cluster_container_buffer = container_buffer<cluster_id, cell>;

/// Convenience declaration for the cluster container view type to use in host
/// code
using cluster_container_view = container_view<cluster_id, cell>;

/// Convenience declaration for the cluster container view type to use in host
/// code (const)
using cluster_container_const_view =
    container_view<const cluster_id, const cell>;

/// Function for signal modelling
inline scalar signal_cell_modelling(scalar signal_in,
                                    const cluster_id& /*cl_id*/) {
    // Retrieve the signal based on the cl_id.module_idx
    return signal_in;
}

/// Function for pixel segmentation
inline vector2 position_from_cell(cell c, cluster_id cl_id) {
    if (cl_id.is_default)
        return {static_cast<scalar>(c.channel0),
                static_cast<scalar>(c.channel1)};
    else
        // Retrieve the specific values based on module idx
        return {cl_id.pixel.min_center_x + c.channel0 * cl_id.pixel.pitch_x,
                cl_id.pixel.min_center_y + c.channel1 * cl_id.pixel.pitch_y};
}

inline vector2 get_pitch(cluster_id cl_id) {
    // return the values based on the module idx
    return {cl_id.pixel.pitch_x, cl_id.pixel.pitch_y};
}

}  // namespace traccc