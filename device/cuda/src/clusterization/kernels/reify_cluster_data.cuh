/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/silicon_cluster_collection.hpp"

namespace traccc::cuda::kernels {
__global__ void reify_cluster_data(
    unsigned int* disjoint_set_ptr, unsigned int num_cells,
    traccc::edm::silicon_cluster_collection::view cluster_view);
}
