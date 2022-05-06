/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"

// Vecmem include(s).
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::sycl {

/// Forward declaration of component connection function
///
void component_connection(
    cluster_container_types::view clusters_view,
    const cell_container_types::host& cells_per_event,
    vecmem::data::vector_view<unsigned int> clusters_count_view,
    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl
