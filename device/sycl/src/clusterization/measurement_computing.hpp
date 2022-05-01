/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/clusterization/measurement_creation_helper.hpp"

// Vecmem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

// Decleration of measurement computing kernel
void measurement_computing(measurement_container_view measurements_view,
                           cluster_container_view clusters_view,
                           const unsigned int& range,
                           queue_wrapper queue);

}  // namespace traccc::sycl
