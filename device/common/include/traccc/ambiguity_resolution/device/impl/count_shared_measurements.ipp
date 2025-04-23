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

TRACCC_HOST_DEVICE inline void count_shared_measurements(
    const global_index_t globalIndex,
    const count_shared_measurements_payload& payload) {

    vecmem::device_vector<const unsigned int> accepted_ids(
        payload.accepted_ids_view);

    if (globalIndex >= accepted_ids.size()) {
        return;
    }

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::device_vector<const unsigned int> n_accepted_tracks_per_measurement(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<unsigned int> n_shared(payload.n_shared_view);

    const unsigned int id = accepted_ids.at(globalIndex);

    if (globalIndex == 0) {
        for (int i = 0; i < n_accepted_tracks_per_measurement.size(); i++) {
            printf("meas_id %lu n_accepted %d a \n", unique_meas[i],
                   n_accepted_tracks_per_measurement[i]);
        }
    }

    for (const auto& meas_id : meas_ids[id]) {

        const auto it = thrust::lower_bound(thrust::seq, unique_meas.begin(),
                                            unique_meas.end(), meas_id);
        const auto unique_meas_idx = static_cast<unsigned int>(
            thrust::distance(unique_meas.begin(), it));

        if (id == 4) {
            printf("unique_meas_idx %d meas id %lu nshared %d \n",
                   unique_meas_idx, meas_id, n_shared.at(id));
        }
        if (n_accepted_tracks_per_measurement.at(unique_meas_idx) > 1) {
            vecmem::device_atomic_ref<unsigned int>(n_shared.at(id))
                .fetch_add(1u);
        }
    }
}

}  // namespace traccc::device
