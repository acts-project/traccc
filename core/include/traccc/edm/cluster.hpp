/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/geometry/pixel_data.hpp"

// System include(s).
#include <cstddef>

namespace traccc {

/// Cluster identifier
///
/// It associates the vector of cells (that make up the cluster) with
/// the detector module, and additional necessary parameters, necessary
/// for making use of the cluster.
///
struct cluster_id {

    event_id event = 0;
    std::size_t module_idx = 0;
    geometry_id module = 0;
    transform3 placement = transform3{};
    scalar threshold = 0.;
    pixel_data pixel;
};

/// Declare all cluster container types
using cluster_container_types = container_types<cluster_id, cell>;

}  // namespace traccc
