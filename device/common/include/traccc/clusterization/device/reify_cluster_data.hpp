/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/concepts/thread_id.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {
template <device::concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE void reify_cluster_data(
    const thread_id_t& thread_id,
    vecmem::data::vector_view<const unsigned int> disjoint_set_view,
    traccc::edm::silicon_cluster_collection::view cluster_view) {

    vecmem::device_vector<const unsigned int> disjoint_set(disjoint_set_view);
    traccc::edm::silicon_cluster_collection::device clusters(cluster_view);
    if (auto idx = thread_id.getGlobalThreadIdX(); idx < disjoint_set.size()) {
        clusters.cell_indices().at(disjoint_set.at(idx)).push_back(idx);
    }
}
}  // namespace traccc::device
