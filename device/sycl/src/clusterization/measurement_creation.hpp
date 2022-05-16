/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/measurement_creation_helper.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::sycl {

/// Forward decleration of measurement creation kernel
///
void measurement_creation(measurement_container_view measurements_view,
                          cluster_container_types::const_view clusters_view,
                          const cell_container_types::const_view& cells_view,
                          queue_wrapper queue);

}  // namespace traccc::sycl