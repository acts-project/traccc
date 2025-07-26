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

__launch_bounds__(512) __global__
    void remove_tracks(device::remove_tracks_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ unsigned int shared_tids[512];
    __shared__ measurement_id_type sh_meas_ids[512];
    __shared__ unsigned int sh_threads[512];
    __shared__ unsigned int n_updating_threads;

    auto threadIndex = threadIdx.x;

    bool is_valid_thread = false;
    bool is_duplicate = true;

    shared_tids[threadIndex] = std::numeric_limits<unsigned int>::max();

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const measurement_id_type> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const unsigned int> n_meas(payload.n_meas_view);
    vecmem::device_vector<const unsigned int> meas_id_to_unique_id(
        payload.meas_id_to_unique_id_view);
    vecmem::jagged_device_vector<const unsigned int> tracks_per_measurement(
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
    vecmem::device_vector<measurement_id_type> meas_to_remove(
        payload.meas_to_remove_view);
    vecmem::device_vector<unsigned int> threads(payload.threads_view);

    const unsigned n_accepted_prev = *(payload.n_accepted);

    __syncthreads();

    if (threadIndex == 0) {
        (*payload.n_accepted) -= *(payload.n_removable_tracks);
        n_updating_threads = 0;
    }

    if (threadIndex < *(payload.n_valid_threads)) {
        sh_meas_ids[threadIndex] = meas_to_remove[threadIndex];
        sh_threads[threadIndex] = threads[threadIndex];
        is_valid_thread = true;
    }

    __syncthreads();

    if (is_valid_thread) {
        const auto id = sh_meas_ids[threadIndex];
        is_duplicate = (threadIndex > 0 && sh_meas_ids[threadIndex - 1] == id);
    }

    bool active = false;
    unsigned int pos1;

    if (!is_duplicate && is_valid_thread) {

        const auto id = sh_meas_ids[threadIndex];
        const auto unique_meas_idx = meas_id_to_unique_id.at(id);

        // If there is only one track associated with measurement, the
        // number of shared measurement can be reduced by one
        const auto& tracks = tracks_per_measurement[unique_meas_idx];
        auto track_status = track_status_per_measurement[unique_meas_idx];

        auto trk_id =
            sorted_ids.at(n_accepted_prev - 1 - sh_threads[threadIndex]);

        unsigned int worst_idx =
            thrust::find(thrust::seq, tracks.begin(), tracks.end(), trk_id) -
            tracks.begin();

        track_status[worst_idx] = 0;

        int n_sharing_tracks = 1;
        for (unsigned int i = threadIndex + 1; i < *(payload.n_valid_threads);
             ++i) {

            if (sh_meas_ids[i] == id && sh_threads[i] != sh_threads[i - 1]) {
                n_sharing_tracks++;

                trk_id = sorted_ids[n_accepted_prev - 1 - sh_threads[i]];

                worst_idx = thrust::find(thrust::seq, tracks.begin(),
                                         tracks.end(), trk_id) -
                            tracks.begin();

                track_status[worst_idx] = 0;

            } else if (sh_meas_ids[i] != id) {
                break;
            }
        }

        vecmem::device_atomic_ref<unsigned int> n_accepted_per_meas(
            n_accepted_tracks_per_measurement.at(
                static_cast<unsigned int>(unique_meas_idx)));
        const unsigned int N_A =
            n_accepted_per_meas.fetch_sub(n_sharing_tracks);

        if (N_A == 1 + n_sharing_tracks) {
            active = true;
            const unsigned int alive_idx =
                thrust::find(thrust::seq, track_status.begin(),
                             track_status.end(), 1) -
                track_status.begin();

            pos1 = atomicAdd(&n_updating_threads, 1);

            shared_tids[pos1] = static_cast<unsigned int>(tracks[alive_idx]);

            auto tid = shared_tids[pos1];

            const auto m_count = static_cast<unsigned int>(thrust::count(
                thrust::seq, meas_ids[tid].begin(), meas_ids[tid].end(), id));

            const unsigned int N_S =
                vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
                    .fetch_sub(m_count);
        }
    }

    __syncthreads();

    if (active) {
        auto tid = shared_tids[pos1];
        bool already_pushed = false;
        for (unsigned int i = 0; i < pos1; ++i) {
            if (shared_tids[i] == tid) {
                already_pushed = true;
                break;
            }
        }

        if (!already_pushed) {

            // Write updated track IDs
            vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
                *(payload.n_updated_tracks));

            const unsigned int pos2 = num_updated_tracks.fetch_add(1);

            updated_tracks[pos2] = tid;
            is_updated[tid] = 1;

            rel_shared.at(tid) = static_cast<traccc::scalar>(n_shared.at(tid)) /
                                 static_cast<traccc::scalar>(n_meas.at(tid));
        }
    }
}

}  // namespace traccc::cuda::kernels
