/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"

// Vecmem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/containers/data/jagged_vector_view.hpp>
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace traccc::sycl {

/// Forward declaration of component connection function
///
void cluster_counting(
    const host_cell_container& cells_per_event,
    vecmem::data::jagged_vector_view<unsigned int> sparse_ccl_indices_view,
    vecmem::data::vector_view<std::size_t> cluster_sizes_view,
    vecmem::data::vector_view<std::size_t> cluster_prefix_sum_view,
    vecmem::data::vector_view<std::size_t> clusters_per_module_view,
    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl