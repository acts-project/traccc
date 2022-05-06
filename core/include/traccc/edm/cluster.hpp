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

/// Header / identifier for every reconstructed cluster
struct cluster_id {

    event_id event{0};
    std::size_t module_idx{0};
    geometry_id module{0};
    transform3 placement{};
    scalar threshold{0.};
    pixel_data pixel{};
};

// Declare all cluster container types
TRACCC_DECLARE_CONTAINER_TYPES(cluster, cluster_id, cell);

}  // namespace traccc
