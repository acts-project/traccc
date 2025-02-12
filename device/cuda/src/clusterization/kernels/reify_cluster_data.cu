/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "../../utils/thread_id.hpp"
#include "reify_cluster_data.cuh"
#include "traccc/clusterization/device/reify_cluster_data.hpp"

namespace traccc::cuda::kernels {
__global__ void reify_cluster_data(
    unsigned int* disjoint_set_ptr, unsigned int num_cells,
    traccc::edm::silicon_cluster_collection::view cluster_view) {
    const details::thread_id1 thread_id;

    device::reify_cluster_data(thread_id, disjoint_set_ptr, num_cells,
                               cluster_view);
}
}  // namespace traccc::cuda::kernels
