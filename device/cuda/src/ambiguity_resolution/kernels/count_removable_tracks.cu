/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/global_index.hpp"
#include "count_removable_tracks.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>

namespace traccc::cuda::kernels {

__device__ void count_tracks(int* sh_n_meas, int n_tracks, int bound,
                             unsigned int& count) {
    // @TODO: Improve the logic
    for (unsigned int stride = 1; stride < n_tracks; stride *= 2) {
        sh_n_meas[threadIdx.x] += sh_n_meas[threadIdx.x + stride];
        __syncthreads();

        if (sh_n_meas[0] < bound) {
            if (threadIdx.x == 0) {
                count += stride;
            }
        }
    }

    __syncthreads();
}

__device__ void bitonic_sort_shared(
    traccc::pair<std::size_t, unsigned int>* shared_data, int count,
    bool compare_first = true) {

    // padding up to next power of 2
    int N = 1;
    while (N < count) {
        N <<= 1;
    }

    // pad unused elements with max value
    int tid = threadIdx.x;
    if (tid >= count && tid < N) {
        shared_data[tid] = {std::numeric_limits<std::size_t>::max(),
                            std::numeric_limits<unsigned int>::max()};
    }

    __syncthreads();

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;

            if (ixj > tid && ixj < N && tid < N) {
                auto elem_i = shared_data[tid];
                auto elem_j = shared_data[ixj];

                bool ascending = ((tid & k) == 0);
                bool should_swap =
                    compare_first
                        ? (elem_i.first > elem_j.first) == ascending
                        : (elem_i.second > elem_j.second) == ascending;

                if (should_swap) {
                    shared_data[tid] = elem_j;
                    shared_data[ixj] = elem_i;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void count_removable_tracks(
    device::count_removable_tracks_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ int shared_n_meas[1024];
    __shared__ traccc::pair<std::size_t, unsigned int> meas_to_thread[1024];
    __shared__ unsigned int n_meas_total;
    __shared__ unsigned int n_tracks_to_iterate;
    __shared__ unsigned int min_thread;

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> n_meas(payload.n_meas_view);
    vecmem::device_vector<traccc::pair<std::size_t, unsigned int>>
        meas_to_remove(payload.meas_to_remove_view);

    auto threadIndex = threadIdx.x;

    int gid = static_cast<int>(*payload.n_accepted) - 1 - threadIndex;
    shared_n_meas[threadIndex] = 0;
    meas_to_thread[threadIndex] = {std::numeric_limits<std::size_t>::max(),
                                   std::numeric_limits<unsigned int>::max()};

    if (threadIndex == 0) {
        *(payload.n_removable_tracks) = 0;
        *(payload.n_meas_to_remove) = 0;
        n_meas_total = 0;
        n_tracks_to_iterate = 0;
        min_thread = std::numeric_limits<unsigned int>::max();
    }

    __syncthreads();

    if (gid >= 0) {
        const auto trk_id = sorted_ids[gid];
        shared_n_meas[threadIndex] = n_meas[trk_id];
    }

    __syncthreads();

    auto n_tracks_total = min(blockDim.x, *payload.n_accepted);

    // @TODO: Improve the logic
    count_tracks(shared_n_meas, n_tracks_total, 1024, n_tracks_to_iterate);

    /*
    if (gid >= 0) {
        const auto trk_id = sorted_ids[gid];
        shared_n_meas[threadIndex] = n_meas[trk_id];
    } 
    */   
    /*
    for (unsigned int stride = 1; stride < n_tracks_total; stride *= 2) {
        shared_n_meas[threadIndex] += shared_n_meas[threadIndex + stride];
        __syncthreads();

        if (shared_n_meas[0] < 1024) {
            if (threadIndex == 0) {
                n_tracks_to_iterate = stride;
            }
        }
    }

    __syncthreads();
    */

    if (threadIndex == 0 && n_tracks_to_iterate == 0) {
        n_tracks_to_iterate = 1;
    }

    __syncthreads();

    // @TODO: Improve the logic
    vecmem::device_atomic_ref<unsigned int> num_meas_total(n_meas_total);

    if (threadIndex < n_tracks_to_iterate && gid >= 0) {
        auto mids = meas_ids[sorted_ids[gid]];
        for (const auto& id : mids) {

            const unsigned int pos = num_meas_total.fetch_add(1);

            meas_to_thread[pos] = {id, threadIndex};
        }
    }

    __syncthreads();

    /*
    if (threadIndex == 0) {
        printf("n meas total %d \n", n_meas_total);
    }
    */

    // Bitonic sort on meas_to_thread w.r.t. measurement id
    bitonic_sort_shared(meas_to_thread, n_meas_total);

    // Find starting point
    if (threadIndex < n_meas_total) {
        auto curr = meas_to_thread[threadIndex];
        bool is_start = (threadIndex == 0) ||
                        (meas_to_thread[threadIndex - 1].first != curr.first);

        // Find min thread id
        std::size_t id = curr.first;
        auto tid = curr.second;

        if (is_start) {

            int i = threadIndex + 1;
            while (i < n_meas_total && meas_to_thread[i].first == id) {
                if (meas_to_thread[i].second != tid) {
                    atomicMin(&min_thread, meas_to_thread[i].second);
                }
                i++;
            }

            /*
            while (i < n_meas_total && (meas_to_thread[i].first == id &&
                                        meas_to_thread[i].second != tid)) {
                atomicMin(&min_thread, meas_to_thread[i].second);
                i++;
            }
            */
        }
    }

    __syncthreads();

    if (threadIndex == 0 &&
        min_thread == std::numeric_limits<unsigned int>::max()) {
        min_thread = 0;
    }

    __syncthreads();

    // Bubble sort w.r.t thread index
    bitonic_sort_shared(meas_to_thread, n_meas_total, false);

    // Make measurement list to remove
    const auto tid = meas_to_thread[threadIndex].second;
    if ((tid < min_thread) || (min_thread == 0 && tid == 0)) {
        meas_to_remove[threadIndex] = meas_to_thread[threadIndex];
        atomicAdd(payload.n_meas_to_remove, 1);
    }

    if (threadIndex == 0) {
        if (min_thread == 0) {
            *(payload.n_removable_tracks) = 1;
        } else {
            *(payload.n_removable_tracks) = min_thread;
        }

        // printf("n removable tracks %d \n", *(payload.n_removable_tracks));
    }
}

}  // namespace traccc::cuda::kernels
