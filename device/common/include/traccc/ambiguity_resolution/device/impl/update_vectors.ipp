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

    __shared__ unsigned int tids[1024];

    if (globalIndex == 0) {
        (*payload.update_res).n_updated_tracks = 0;
        /*
        printf("n accepted %d \n", *payload.n_accepted == 0);
        printf("Max shared %d n accepted %d n updated %d \n",
               (*payload.update_res).max_shared,
               (*payload.update_res).n_accepted,
               (*payload.update_res).n_updated_tracks);
        */
    }

    tids[globalIndex] = std::numeric_limits<unsigned int>::max();

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

    if (*payload.n_accepted == 0) {
        return;
    }

    const auto worst_track = sorted_ids[*payload.n_accepted - 1];
    const auto& meas_ids_of_track = meas_ids[worst_track];
    /*
    if (globalIndex == 0) {

        printf("Sorted ids: ");
        for (int i = 0; i < *payload.n_accepted; i++) {
            printf("(%d %f)", sorted_ids[i], rel_shared[sorted_ids[i]]);
        }
        printf("\n");

        printf("Worst track: %d \n", worst_track);
    }
    */
    if (globalIndex == 0) {
        (*payload.n_accepted)--;
        (*payload.update_res).n_accepted = *payload.n_accepted;
    }

    barrier.blockBarrier();

    if (globalIndex >= n_meas[worst_track])
        return;

    const auto id = meas_ids_of_track[globalIndex];

    if (thrust::find(thrust::seq, meas_ids_of_track.begin(),
                     meas_ids_of_track.begin() + globalIndex,
                     id) != meas_ids_of_track.begin() + globalIndex) {
        return;
    }

    const std::size_t unique_meas_idx =
        thrust::lower_bound(thrust::seq, unique_meas.begin(), unique_meas.end(),
                            id) -
        unique_meas.begin();

    vecmem::device_atomic_ref<unsigned int> n_accepted_per_meas(
        n_accepted_tracks_per_measurement.at(
            static_cast<unsigned int>(unique_meas_idx)));
    const unsigned int N_A = n_accepted_per_meas.fetch_add(-1u);

    // If there is only one track associated with measurement, the
    // number of shared measurement can be reduced by one
    const auto& tracks = tracks_per_measurement[unique_meas_idx];
    auto track_status = track_status_per_measurement[unique_meas_idx];

    const unsigned int worst_idx =
        thrust::find(thrust::seq, tracks.begin(), tracks.end(), worst_track) -
        tracks.begin();

    track_status[worst_idx] = 0;

    if (N_A != 2) {
        return;
    }

    // printf("threadIdx.x %d meas id %lu \n", threadIdx.x, id);

    const unsigned int alive_idx =
        thrust::find(thrust::seq, track_status.begin(), track_status.end(), 1) -
        track_status.begin();

    tids[threadIdx.x] = static_cast<unsigned int>(tracks[alive_idx]);
    auto tid = tids[threadIdx.x];

    /*
    if ((*payload.update_res).max_shared == 0) {
        return;
    }
    */
    barrier.blockBarrier();

    /*
    if (threadIdx.x > 0 && thrust::find(thrust::seq, tids, tids + threadIdx.x,
                                        tid) != tids + threadIdx.x) {
        return;
    }
    */
    const auto m_count = static_cast<unsigned int>(thrust::count(
        thrust::seq, meas_ids[tid].begin(), meas_ids[tid].end(), id));

    const unsigned int N_S =
        vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
            .fetch_sub(m_count);

    rel_shared.at(tid) = static_cast<traccc::scalar>(n_shared.at(tid)) /
                         static_cast<traccc::scalar>(n_meas.at(tid));

    if (threadIdx.x == 0 || thrust::find(thrust::seq, tids, tids + threadIdx.x,
                                         tid) == tids + threadIdx.x) {

        // Write updated track IDs
        vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
            (*payload.update_res).n_updated_tracks);

        const unsigned int pos = num_updated_tracks.fetch_add(1);

        updated_tracks[pos] = tid;
        /*
        printf("Added updated tid: %d %d %d measid %lu \n", tid, m_count,
               n_shared.at(tid), id);
        */
    }
    /*
    if (thrust::find(thrust::seq, updated_tracks.begin(),
                     updated_tracks.begin() + pos,
                     tid) != updated_tracks.begin() + pos) {
        printf("Found updated tid: %d \n", tid);
        num_updated_tracks.fetch_sub(1);
    } else {
        printf("Added updated tid: %d \n", tid);
        updated_tracks[pos] = tid;
    }
    */
}

}  // namespace traccc::device
