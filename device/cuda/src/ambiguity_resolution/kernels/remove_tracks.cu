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

__device__ void count_tracks(int tid, int* sh_n_meas, int n_tracks,
                             unsigned int& bound, unsigned int& count,
                             bool& stop) {

    unsigned int add = 0;
    unsigned int offset = 0;
    for (unsigned int stride = 1; stride < (n_tracks - count); stride *= 2) {
        if ((count + tid + stride) < n_tracks) {
            sh_n_meas[count + tid] += sh_n_meas[count + tid + stride];
        }
        __syncthreads();

        if (sh_n_meas[count] < bound) {
            if (tid == 0) {
                offset = sh_n_meas[count];
                add = stride * 2;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        bound -= offset;
        count += add;

        if (add == 0) {
            stop = true;
        }
    }

    __syncthreads();
}

__launch_bounds__(512) __global__
    void remove_tracks(device::remove_tracks_payload payload) {

    if (threadIdx.x == 0) {
        if (*(payload.max_shared) == 0) {
            *(payload.terminate) = 1;
        }
    }

    __syncthreads();

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ int sh_buffer[512];
    __shared__ measurement_id_type sh_meas_ids[512];
    __shared__ unsigned int sh_threads[512];
    __shared__ unsigned int n_meas_total;
    __shared__ unsigned int bound;
    __shared__ unsigned int n_tracks_to_iterate;
    __shared__ unsigned int min_thread;
    __shared__ unsigned int N;
    __shared__ bool stop;
    __shared__ unsigned int n_updating_threads;

    auto threadIndex = threadIdx.x;

    int gid = static_cast<int>(*payload.n_accepted) - 1 - threadIndex;
    sh_buffer[threadIndex] = 0;
    sh_meas_ids[threadIndex] = std::numeric_limits<measurement_id_type>::max();
    sh_threads[threadIndex] = std::numeric_limits<unsigned int>::max();

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

    if (threadIndex == 0) {
        *(payload.n_removable_tracks) = 0;
        *(payload.n_meas_to_remove) = 0;
        *(payload.n_valid_threads) = 0;
        n_meas_total = 0;
        bound = 512;
        N = 1;
        n_tracks_to_iterate = 0;
        min_thread = std::numeric_limits<unsigned int>::max();
        stop = false;
    }

    __syncthreads();

    unsigned int trk_id = 0;
    unsigned int n_m = 0;
    if (gid >= 0) {
        trk_id = sorted_ids[gid];
        n_m = n_meas[trk_id];

        // Buffer for the number of measurement per track
        sh_buffer[threadIndex] = n_m;
    }

    __syncthreads();

    auto n_tracks_total = min(bound, *payload.n_accepted);

    /****************************************
     * Count the number of removable tracks
     ****************************************/

    // @TODO: Improve the logic
    count_tracks(threadIdx.x, sh_buffer, n_tracks_total, bound,
                 n_tracks_to_iterate, stop);
    /*
    for (int i = 0; i < 100; i++) {
        count_tracks(threadIdx.x, shared_n_meas, n_tracks_total, bound,
                     n_tracks_to_iterate, stop);
        __syncthreads();
        if (stop)
            break;

        if (gid >= 0 && static_cast<unsigned int>(gid) < sorted_ids.size()) {
            const auto trk_id = sorted_ids[gid];
            if (trk_id < n_meas.size()) {
                shared_n_meas[threadIndex] = n_meas[trk_id];
            }
        }
        __syncthreads();
    }
    */

    if (threadIndex == 0 && n_tracks_to_iterate == 0) {
        n_tracks_to_iterate = 1;
    }

    // @TODO: Improve the logic
    if (threadIndex < n_tracks_to_iterate && gid >= 0) {
        const unsigned int pos = atomicAdd(&n_meas_total, n_m);

        const auto& mids = meas_ids[trk_id];
        for (int i = 0; i < n_m; i++) {
            sh_meas_ids[pos + i] = mids[i];
            sh_threads[pos + i] = threadIndex;
        }
    }

    __syncthreads();

    // Bitonic sort on meas_to_thread w.r.t. measurement id
    if (threadIndex == 0) {
        N = (n_meas_total == 0) ? 1 : 1 << (32 - __clz(n_meas_total - 1));
    }
    __syncthreads();

    const auto tid = threadIndex;
    for (int k = 2; k <= N; k <<= 1) {

        bool ascending = ((tid & k) == 0);

        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;

            if (ixj > tid && ixj < N && tid < N) {
                auto meas_i = sh_meas_ids[tid];
                auto meas_j = sh_meas_ids[ixj];
                auto thread_i = sh_threads[tid];
                auto thread_j = sh_threads[ixj];

                bool should_swap =
                    (meas_i > meas_j ||
                     (meas_i == meas_j && thread_i > thread_j)) == ascending;

                if (should_swap) {
                    sh_meas_ids[tid] = meas_j;
                    sh_meas_ids[ixj] = meas_i;
                    sh_threads[tid] = thread_j;
                    sh_threads[ixj] = thread_i;
                }
            }
            __syncthreads();
        }
    }

    // Find starting point
    if (threadIndex < n_meas_total) {
        auto mid = sh_meas_ids[threadIndex];
        bool is_start =
            (threadIndex == 0) || (sh_meas_ids[threadIndex - 1] != mid);
        const auto unique_meas_idx = meas_id_to_unique_id.at(mid);
        const auto its_accepted_tracks =
            n_accepted_tracks_per_measurement.at(unique_meas_idx);

        if (is_start) {

            int i = threadIndex + 1;
            int n_sharing_tracks = 1;

            while (i < n_meas_total && sh_meas_ids[i] == mid) {
                if (sh_threads[i] != sh_threads[i - 1]) {
                    n_sharing_tracks++;

                    if (n_sharing_tracks == its_accepted_tracks) {
                        atomicMin(&min_thread, sh_threads[i - 1]);
                        break;
                    }
                }
                i++;
            }
        }
    }

    __syncthreads();

    if (threadIndex == 0) {
        if (min_thread == 0) {
            *(payload.n_removable_tracks) = 1;
        } else if (min_thread == std::numeric_limits<unsigned int>::max()) {
            *(payload.n_removable_tracks) = n_tracks_to_iterate;
        } else {
            *(payload.n_removable_tracks) = min_thread;
        }
    }

    __syncthreads();

    auto meas_to_remove_temp = sh_meas_ids[threadIndex];
    auto threads_temp = sh_threads[threadIndex];

    __syncthreads();

    if (threadIndex == 0) {
        *(payload.n_meas_to_remove) = n_meas_total;
    }

    __syncthreads();

    int is_valid = (threads_temp < *(payload.n_removable_tracks)) ? 1 : 0;

    // TODO: Use better reduction algorithm
    if (is_valid) {
        atomicAdd(payload.n_valid_threads, 1);
    }

    __syncthreads();

    // Exclusive scan (Hillis-Steele)

    // Buffer for the prefix
    sh_buffer[threadIndex] = is_valid;  // copy input
    __syncthreads();

    for (int offset = 1; offset < *(payload.n_meas_to_remove); offset <<= 1) {
        int val = 0;
        if (threadIndex >= offset) {
            val = sh_buffer[threadIndex - offset];
        }
        __syncthreads();
        sh_buffer[threadIndex] += val;
        __syncthreads();
    }

    if (is_valid) {
        sh_buffer[threadIndex] -= 1;
        sh_meas_ids[sh_buffer[threadIndex]] = meas_to_remove_temp;
        sh_threads[sh_buffer[threadIndex]] = threads_temp;
    }

    __syncthreads();

    meas_to_remove_temp = sh_meas_ids[threadIndex];
    threads_temp = sh_threads[threadIndex];

    /********************
     * Remove tracks
     ********************/

    __syncthreads();

    bool is_valid_thread = false;
    bool is_duplicate = true;

    const unsigned n_accepted_prev = *(payload.n_accepted);

    __syncthreads();

    if (threadIndex == 0) {
        (*payload.n_accepted) -= *(payload.n_removable_tracks);
        n_updating_threads = 0;
    }

    if (threadIndex < *(payload.n_valid_threads)) {
        sh_meas_ids[threadIndex] = meas_to_remove_temp;
        sh_threads[threadIndex] = threads_temp;
        is_valid_thread = true;
    }

    __syncthreads();

    if (is_valid_thread) {
        const auto id = sh_meas_ids[threadIndex];
        is_duplicate = (threadIndex > 0 && sh_meas_ids[threadIndex - 1] == id);
    }

    // Buffer for the track ids
    sh_buffer[threadIndex] = std::numeric_limits<int>::max();

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
            thrust::lower_bound(thrust::seq, tracks.begin(), tracks.end(),
                                trk_id) -
            tracks.begin();

        track_status[worst_idx] = 0;

        int n_sharing_tracks = 1;
        for (unsigned int i = threadIndex + 1; i < *(payload.n_valid_threads);
             ++i) {

            if (sh_meas_ids[i] == id && sh_threads[i] != sh_threads[i - 1]) {
                n_sharing_tracks++;

                trk_id = sorted_ids[n_accepted_prev - 1 - sh_threads[i]];

                worst_idx = thrust::lower_bound(thrust::seq, tracks.begin(),
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

            sh_buffer[pos1] = static_cast<int>(tracks[alive_idx]);

            auto tid = sh_buffer[pos1];

            const auto m_count = static_cast<unsigned int>(thrust::count(
                thrust::seq, meas_ids[tid].begin(), meas_ids[tid].end(), id));

            const unsigned int N_S =
                vecmem::device_atomic_ref<unsigned int>(n_shared.at(tid))
                    .fetch_sub(m_count);
        }
    }

    __syncthreads();

    if (active) {
        auto tid = sh_buffer[pos1];
        bool already_pushed = false;
        for (unsigned int i = 0; i < pos1; ++i) {
            if (sh_buffer[i] == tid) {
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
