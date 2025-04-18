/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void update_tracks_per_measurement(
    const global_index_t globalIndex,
    const update_tracks_per_measurement_payload& payload) {

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::device_vector<unsigned int> n_accepted_tracks_per_measurement(
        payload.n_accepted_tracks_per_measurement_view);

    const auto id = meas_ids[payload.worst_track][globalIndex];

    const auto it = thrust::lower_bound(thrust::seq, unique_meas.begin(),
                                        unique_meas.end(), id);
    const std::size_t unique_meas_idx =
        static_cast<std::size_t>(thrust::distance(unique_meas.begin(), it));

    vecmem::device_atomic_ref<unsigned int> n_accepted(
        n_accepted_tracks_per_measurement.at(
            static_cast<unsigned int>(unique_meas_idx)));
    n_accepted.fetch_add(-1u);
}

}  // namespace traccc::device
