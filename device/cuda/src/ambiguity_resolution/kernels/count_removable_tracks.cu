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

__global__ void count_removable_tracks(
    device::count_removable_tracks_payload payload) {

    if (*(payload.terminate) == 1) {
        return;
    }

    __shared__ int shared_n_meas[1024];
    __shared__ traccc::pair<std::size_t, unsigned int> meas_to_thread[1024];
    __shared__ unsigned int n_meas_total;
    __shared__ unsigned int n_tracks;

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::jagged_device_vector<const std::size_t> meas_ids(
        payload.meas_ids_view);
    vecmem::device_vector<const std::size_t> n_meas(payload.n_meas_view);

    auto threadIndex = threadIdx.x;
    int gid = static_cast<int>(*payload.n_accepted) - 1 - threadIndex;

    if (gid < 0) {
        return;
    }
    const auto trk_id = sorted_ids[*payload.n_accepted - 1 - threadIndex];
    shared_n_meas[threadIndex] = n_meas[trk_id];

    __syncthreads();

    auto blockSize = blockDim.x;

    // @TODO: Improve the logic
    for (unsigned int stride = 1; stride < blockSize; stride *= 2) {
        // auto temp = shared_n_meas[threadIndex];

        shared_n_meas[threadIndex] += shared_n_meas[threadIndex + stride];
        __syncthreads();

        if (shared_n_meas[0] > 1024) {
            if (threadIndex == 0) {
                // n_meas_total = temp;
                n_tracks = stride;
            }
        }
    }

    __syncthreads();

    if (threadIndex == 0) {
        n_meas_total = 0;
    }

    // @TODO: Improve the logic
    if (threadIndex < n_tracks) {
        auto mids = meas_ids[threadIndex];
        for (const auto& id : mids) {

            // Write updated track IDs
            vecmem::device_atomic_ref<unsigned int> num_meas_total(
                n_meas_total);

            const unsigned int pos = num_meas_total.fetch_add(1);

            meas_to_thread[pos] = {id, threadIndex};
        }
    }

    __syncthreads();

    (void)meas_to_thread;

    printf("%d \n", meas_to_thread[threadIndex].second);
    /*
    // Bitonic sort
    for (int k = 2; k <= n_meas_total; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ix = threadIdx.x;
            if (ix < n) {
                int ixj = ix ^ j;
                if (ixj < n) {
                    auto a = meas_to_thread[ix];
                    auto b = meas_to_thread[ixj];
                    bool should_swap = (ix < ixj) == (a.first > b.first);
                    if (should_swap) {
                        meas_to_thread[ix] = b;
                        meas_to_thread[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    */
    // Make measurement list to remove

    /*
    if (!any_overlap) {
        const unsigned int N_m =
            vecmem::device_atomic_ref<unsigned int>(n_worst_tracks)
                .fetch_add(1);
    }
    */
    // Sort measurement list
}

}  // namespace traccc::cuda::kernels
