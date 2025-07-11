/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../../utils/barrier.hpp"
#include "../../utils/global_index.hpp"
#include "sort_updated_tracks.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

__global__ void sort_updated_tracks(
    device::sort_updated_tracks_payload payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    extern __shared__ unsigned int shared_mem_tracks[];

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int N = *(payload.n_updated_tracks);

    // Load updated track indices into shared memory (for sorting)
    if (tid < N) {
        shared_mem_tracks[tid] = updated_tracks[tid];
    }

    __syncthreads();

    // odd-even sort
    // @TODO: Use bitonic sort instead, which is faster
    for (int iter = 0; iter < N; ++iter) {
        bool is_even = (iter % 2 == 0);
        int i = tid;

        if (i < N / 2) {
            int idx = 2 * i + (is_even ? 0 : 1);
            if (idx + 1 < N) {
                unsigned int a = shared_mem_tracks[idx];
                unsigned int b = shared_mem_tracks[idx + 1];

                traccc::scalar rel_a = rel_shared[a];
                traccc::scalar rel_b = rel_shared[b];
                traccc::scalar pv_a = pvals[a];
                traccc::scalar pv_b = pvals[b];

                bool swap = false;
                if (rel_a != rel_b) {
                    swap = rel_a > rel_b;
                } else {
                    swap = pv_a < pv_b;
                }

                if (swap) {
                    shared_mem_tracks[idx] = b;
                    shared_mem_tracks[idx + 1] = a;
                }
            }
        }
        __syncthreads();
    }

    // Write back the sorted result from shared memory to global memory
    if (tid < N) {
        updated_tracks[tid] = shared_mem_tracks[tid];
    }
}

}  // namespace traccc::cuda::kernels
