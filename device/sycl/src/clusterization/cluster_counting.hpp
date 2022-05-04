/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/detail/sparse_ccl.hpp"
#include "traccc/edm/cell.hpp"

// Vecmem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/memory/unique_ptr.hpp>

namespace traccc::sycl {

/// Forward declaration of component connection function
///
void cluster_counting(
    const host_cell_container& cells_per_event,
    vecmem::unique_alloc_ptr<unsigned int>& cluster_sum,
    vecmem::unique_alloc_ptr<unsigned int>& cluster_max,
    vecmem::memory_resource& resource, queue_wrapper queue);

}  // namespace traccc::sycl