/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"

namespace traccc::device {
template <device::concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE void reify_cluster_data(
    const thread_id_t& thread_id, unsigned int* disjoint_set_ptr,
    unsigned int num_cells,
    traccc::edm::silicon_cluster_collection::view cluster_view) {
    traccc::edm::silicon_cluster_collection::device clusters(cluster_view);
    if (auto idx = thread_id.getGlobalThreadIdX(); idx < num_cells) {
        clusters.cell_indices().at(disjoint_set_ptr[idx]).push_back(idx);
    }
}
}  // namespace traccc::device
