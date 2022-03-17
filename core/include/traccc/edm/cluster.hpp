/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"

// System include(s).
#include <functional>
#include <vector>

namespace traccc {

using position_estimation = std::function<vector2(channel_id, channel_id)>;
using signal_modeling = std::function<scalar(scalar)>;

struct cluster_id {

    event_id event = 0;
    geometry_id module = 0;
    transform3 placement = transform3{}; 

    position_estimation position_from_cell = [](channel_id ch0,
                                                channel_id ch1) -> vector2 {
        return {static_cast<scalar>(ch0), static_cast<scalar>(ch1)};
    };

    scalar threshold = 0.;
    signal_modeling signal = [](scalar signal_in) -> scalar {
        return signal_in;
    };

};

/// Convenience declaration for the cluster container type to use in host code
using host_cluster_container = host_container<cluster_id, cell>;

/// Convenience declaration for the cluster container type to use in device code
using device_cluster_container = device_container<cluster_id, cell>;

/// Convenience declaration for the cluster container data type to use in host code
using cluster_container_data = container_data<cluster_id, cell>;

/// Convenience declaration for the cluster container buffer type to use in host
/// code
using cluster_container_buffer = container_buffer<cluster_id, cell>;

/// Convenience declaration for the cluster container view type to use in host code
using cluster_container_view = container_view<cluster_id, cell>;

/// A cluster definition:
///
/// a list of cells that make up the cluster
struct cluster {
    std::vector<cell> cells;
};

/// A cluster collection which carries the geometry_id, the clusters
/// and the additional information to create the cluster position
/// from the channel id;
struct cluster_collection {

    event_id event = 0;
    geometry_id module = 0;
    transform3 placement = transform3{};

    std::vector<cluster> items;

    position_estimation position_from_cell = [](channel_id ch0,
                                                channel_id ch1) -> vector2 {
        return {static_cast<scalar>(ch0), static_cast<scalar>(ch1)};
    };

    scalar threshold = 0.;
    signal_modeling signal = [](scalar signal_in) -> scalar {
        return signal_in;
    };
};

using cluster_container = std::vector<cluster_collection>;
}  // namespace traccc
