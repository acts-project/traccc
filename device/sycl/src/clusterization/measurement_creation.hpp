/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc::sycl {

/// Forward decleration of measurement creation kernel
///
void measurement_creation(measurement_container_view measurements_view,
                          cluster_container_types::const_view clusters_view,
                          queue_wrapper queue);

}  // namespace traccc::sycl
