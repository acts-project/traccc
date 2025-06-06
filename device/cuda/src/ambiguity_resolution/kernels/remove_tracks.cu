/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/utils/pair.hpp"

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "count_removable_tracks.cuh"
#include "remove_tracks.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::cuda::kernels {

__global__ void remove_tracks(device::remove_tracks_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ unsigned int shared_tids[1024];
    __shared__ std::size_t sh_meas_ids[1024];
    __shared__ unsigned int sh_threads[1024];

    auto threadIndex = threadIdx.x;

    shared_tids[threadIndex] = std::numeric_limits<unsigned int>::max();

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
    vecmem::device_vector<int> is_updated(payload.is_updated_view);
    vecmem::device_vector<std::size_t> meas_to_remove(
        payload.meas_to_remove_view);
    vecmem::device_vector<unsigned int> threads(payload.threads_view);

    auto n_accepted_prev = (*payload.n_accepted);
    if (threadIndex == 0) {
        (*payload.n_accepted) -= *(payload.n_removable_tracks);
    }

    if (threadIndex < *(payload.n_meas_to_remove)) {
        sh_meas_ids[threadIndex] = meas_to_remove[threadIndex];
        sh_threads[threadIndex] = threads[threadIndex];
    } else {
        return;
    }

    const auto id = sh_meas_ids[threadIndex];

    bool is_duplicate = false;
    for (unsigned int i = 0; i < threadIndex; ++i) {
        if (sh_meas_ids[i] == id) {
            is_duplicate = true;
            break;
        }
    }
    if (is_duplicate) {
        return;
    }

    const std::size_t unique_meas_idx =
        thrust::lower_bound(thrust::seq, unique_meas.begin(), unique_meas.end(),
                            id) -
        unique_meas.begin();

    // If there is only one track associated with measurement, the
    // number of shared measurement can be reduced by one
    const auto& tracks = tracks_per_measurement[unique_meas_idx];
    auto track_status = track_status_per_measurement[unique_meas_idx];

    auto trk_id = sorted_ids[n_accepted_prev - 1 - sh_threads[threadIndex]];

    unsigned int worst_idx =
        thrust::find(thrust::seq, tracks.begin(), tracks.end(), trk_id) -
        tracks.begin();

    track_status[worst_idx] = 0;

    int n_sharing_tracks = 1;
    for (unsigned int i = threadIndex + 1; i < *(payload.n_meas_to_remove);
         ++i) {

        if (sh_meas_ids[i] == id && sh_threads[i] != sh_threads[i - 1]) {
            n_sharing_tracks++;

            trk_id = sorted_ids[n_accepted_prev - 1 - sh_threads[i]];

            worst_idx = thrust::find(thrust::seq, tracks.begin(), tracks.end(),
                                     trk_id) -
                        tracks.begin();

            track_status[worst_idx] = 0;

        } else if (sh_meas_ids[i] != id) {
            break;
        }
    }

    vecmem::device_atomic_ref<unsigned int> n_accepted_per_meas(
        n_accepted_tracks_per_measurement.at(
            static_cast<unsigned int>(unique_meas_idx)));
    const unsigned int N_A = n_accepted_per_meas.fetch_sub(n_sharing_tracks);

    if (N_A != 1 + n_sharing_tracks) {
        return;
    }

    const unsigned int alive_idx =
        thrust::find(thrust::seq, track_status.begin(), track_status.end(), 1) -
        track_status.begin();

    shared_tids[threadIndex] = static_cast<unsigned int>(tracks[alive_idx]);

    __syncthreads();

    auto tid = shared_tids[threadIndex];

    const auto m_count = static_cast<unsigned int>(thrust::count(
        thrust::seq, meas_ids[tid].begin(), meas_ids[tid].end(), id));

    const unsigned int N_S =
        vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
            .fetch_sub(m_count);

    bool already_pushed = false;
    for (unsigned int i = 0; i < threadIndex; ++i) {
        if (shared_tids[i] == tid) {
            already_pushed = true;
            break;
        }
    }
    if (!already_pushed) {

        // Write updated track IDs
        vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
            *(payload.n_updated_tracks));

        const unsigned int pos = num_updated_tracks.fetch_add(1);

        updated_tracks[pos] = tid;
        is_updated[tid] = 1;

        rel_shared.at(tid) = static_cast<traccc::scalar>(n_shared.at(tid)) /
                             static_cast<traccc::scalar>(n_meas.at(tid));
    }
}

}  // namespace traccc::cuda::kernels
