/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void find_lower_bounds_payload(
    const global_index_t globalIndex,
    const find_lower_bounds_payload& payload) {

    if (globalIndex >= payload.n_updated_tracks) {
        return;
    }

    vecmem::device_vector<const std::size_t> updated_trakcs(
        payload.updated_tracks_view);
    vecmem::device_vector<const std::size_t> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const std::size_t> lower_bounds(
        payload.lower_bounds_view);

    const auto tid = updated_tracks[globalIndex];

    const auto it = std::find(sorted_ids.begin(), sorted_ids.end(), tid);

    const auto it2 = thrust::lower_bound(thrust::seq, sorted_ids.begin(), it,
                                         tid, track_comparator);
}

}  // namespace traccc::device
