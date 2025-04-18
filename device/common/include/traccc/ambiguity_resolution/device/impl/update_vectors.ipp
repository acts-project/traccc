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

TRACCC_HOST_DEVICE inline void update_vectors(
    const global_index_t globalIndex, const update_vectors_payload& payload) {

    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);

    if (globalIndex >= meas_ids[payload.worst_track].size()) {
        return;
    }

    vecmem::device_vector<const std::size_t> n_meas(payload.n_meas_view);
    vecmem::device_vector<const std::size_t> unique_meas(
        payload.unique_meas_view);
    vecmem::jagged_device_vector<const std::size_t> tracks_per_measurement(
        payload.tracks_per_measurement_view);
    vecmem::jagged_device_vector<int> track_status_per_measurement(
        payload.track_status_per_measurement_view);
    vecmem::device_vector<unsigned int> n_accepted_tracks_per_measurement(
        payload.n_accepted_tracks_per_measurement_view);
    vecmem::device_vector<unsigned int> n_shared(payload.n_shared_view);
    vecmem::device_vector<traccc::scalar> rel_shared(payload.rel_shared_view);

    const auto id = meas_ids[payload.worst_track][globalIndex];

    const auto it = thrust::lower_bound(thrust::seq, unique_meas.begin(),
                                        unique_meas.end(), id);
    const std::size_t unique_meas_idx =
        static_cast<std::size_t>(thrust::distance(unique_meas.begin(), it));

    vecmem::device_atomic_ref<unsigned int> n_accepted(
        n_accepted_tracks_per_measurement.at(
            static_cast<unsigned int>(unique_meas_idx)));
    const unsigned int N_A = n_accepted.fetch_add(-1u);

    // If there is only one track associated with measurement, the
    // number of shared measurement can be reduced by one
    const auto& tracks = tracks_per_measurement[unique_meas_idx];
    printf("n accepted: %d meas id: %lu tid: %d \n", N_A, id,
           static_cast<unsigned int>(tracks[0]));

    const auto it2 = thurst::find(tracks.begin(), tracks.end(), worst_track);
    const std::size_t worst_idx =
        static_cast<std::size_t>(thrust::distance(tracks.begin(), it));

    if (N_A == 2) {
        const auto tid = static_cast<unsigned int>(tracks[worst_idx]);
        const unsigned int N_S =
            vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
                .fetch_add(-1u);
        printf("n s: %d meas id: %lu tid: %d \n", N_S, id,
               static_cast<unsigned int>(tracks[0]));

        /*
        rel_shared[tid] = static_cast<traccc::scalar>(n_shared[tid]) /
                          static_cast<traccc::scalar>(n_meas[tid]);
        */
    }
}

}  // namespace traccc::device
