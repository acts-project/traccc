/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void update_vectors(
    const global_index_t globalIndex, const barrier_t& barrier,
    const update_vectors_payload& payload) {

    const int warp_size = 32;
    const int warps_per_block = 32;

    __shared__ unsigned int shared_per_warp[warps_per_block];
    __shared__ unsigned int tid_per_warp[warps_per_block];

    if (globalIndex == 0) {
        *payload.n_updated_tracks = 0;
        *payload.max_shared = 0;
    }

    barrier.blockBarrier();

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
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
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    // Find max shared
    const auto n_iter = payload.n_accepted / blockDim.x + 1;
    auto max_track_id = 0;
    unsigned int warp_id = threadIdx.x / warp_size;

    for (int i = 0; i < n_iter; i++) {
        const auto gid = globalIndex + i * blockDim.x;

        unsigned int tid = 0;
        unsigned int shared = 0;

        if (gid < payload.n_accepted) {
            tid = sorted_ids[gid];
            shared = n_shared[tid];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned int other_tid = __shfl_down_sync(0xffffffff, tid, offset);
            unsigned int other_shared =
                __shfl_down_sync(0xffffffff, shared, offset);

            if (other_shared > shared) {
                tid = other_tid;
                shared = other_shared;
            }
        }

        if (gid % warp_size == 0) {
            shared_per_warp[warp_id] = shared;
            tid_per_warp[warp_id] = tid;
        }

        barrier.blockBarrier();

        if (globalIndex == 0) {
            for (int i = 0; i < warps_per_block; ++i) {
                if (shared_per_warp[i] > *payload.max_shared) {
                    *payload.max_shared = shared_per_warp[i];
                    max_track_id = tid_per_warp[i];
                }
            }
        }
    }

    const auto worst_track = sorted_ids[payload.n_accepted - 1];
    const auto& meas_ids_of_track = meas_ids[worst_track];

    if (globalIndex >= meas_ids_of_track.size()) {
        return;
    }

    const auto id = meas_ids_of_track[globalIndex];

    if (thrust::find(thrust::seq, meas_ids_of_track.begin(),
                     meas_ids_of_track.begin() + globalIndex,
                     id) != (meas_ids_of_track.begin() + globalIndex)) {
        return;
    }

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
    auto track_status = track_status_per_measurement[unique_meas_idx];

    const auto it2 =
        thrust::find(thrust::seq, tracks.begin(), tracks.end(), worst_track);
    const unsigned int worst_idx =
        static_cast<unsigned int>(thrust::distance(tracks.begin(), it2));
    track_status[worst_idx] = 0;

    if (N_A == 2) {
        const auto it3 = thrust::find(thrust::seq, track_status.begin(),
                                      track_status.end(), 1);
        const unsigned int alive_idx = static_cast<unsigned int>(
            thrust::distance(track_status.begin(), it3));
        const auto tid = static_cast<unsigned int>(tracks[alive_idx]);

        const unsigned int N_S =
            vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
                .fetch_add(-static_cast<unsigned int>(
                    thrust::count(thrust::seq, meas_ids.at(tid).begin(),
                                  meas_ids.at(tid).end(), id)));

        rel_shared.at(tid) = static_cast<traccc::scalar>(n_shared.at(tid)) /
                             static_cast<traccc::scalar>(n_meas.at(tid));

        // Write updated track IDs
        vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
            *payload.n_updated_tracks);

        const unsigned int pos = num_updated_tracks.fetch_add(1);
        updated_tracks.at(pos) = tid;
    }
}

}  // namespace traccc::device
