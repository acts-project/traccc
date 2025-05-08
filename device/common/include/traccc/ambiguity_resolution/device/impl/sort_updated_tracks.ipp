/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

// Thrust unique(s).
#include <thrust/unique.h>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void sort_updated_tracks(
    const global_index_t globalIndex, const barrier_t& barrier,
    const sort_updated_tracks_payload& payload) {

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    extern __shared__ unsigned int shared_mem_tracks[];

    const unsigned int tid = globalIndex;
    const unsigned int N = (*payload.update_res).n_updated_tracks;

    // Load to shared memory
    if (tid < N) {
        shared_mem_tracks[tid] = updated_tracks[tid];
    }

    barrier.blockBarrier();

    /*    
    if (threadIdx.x == 0) {
        printf("Number of updated tracks %d \n", N);
        printf("Before Sorted updated tracks: ");
        for (int i = 0; i < N; i++) {
            printf("(%d %f)", updated_tracks[i], rel_shared[updated_tracks[i]]);
        }
        printf("\n");
    }
    */

    // Bubble sort
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            unsigned int a = shared_mem_tracks[i];
            unsigned int b = shared_mem_tracks[j];

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
                shared_mem_tracks[i] = b;
                shared_mem_tracks[j] = a;
            }
        }
    }

    barrier.blockBarrier();

    // Write back
    if (tid < N) {
        updated_tracks[tid] = shared_mem_tracks[tid];
    }

    barrier.blockBarrier();

    /*
    if (threadIdx.x == 0) {
        printf("Number of updated tracks %d \n", N);
        printf("After Sorted updated tracks: ");
        for (int i = 0; i < N; i++) {
            printf("(%d %f)", updated_tracks[i], rel_shared[updated_tracks[i]]);
        }
        printf("\n");
    }
    */
   
    barrier.blockBarrier();

    /*
    // Deduplicate (Make it better later)
    if (threadIdx.x == 0) {
        auto end = thrust::unique(thrust::seq, updated_tracks.begin(),
                                  updated_tracks.begin() + N);
        (*payload.update_res).n_updated_tracks = end - updated_tracks.begin();
    }


    if (threadIdx.x == 0) {
        printf("Number of deduplicated tracks %d \n",
               (*payload.update_res).n_updated_tracks);
        printf("After Deduplicate: ");
        for (int i = 0; i < (*payload.update_res).n_updated_tracks; i++) {
            printf("%d ", updated_tracks[i]);
        }
        printf("\n");
    }
    */
    /*
    if (threadIdx.x == 0) {
        (*payload.update_res).n_updated_tracks = 0;
    }

    // === Deduplication ===
    // compact to front if it's the first appearance
    if (tid < N) {
        unsigned int val = shared_mem_tracks[tid];
        unsigned int flag = 1;

        if (tid > 0 && shared_mem_tracks[tid] == shared_mem_tracks[tid - 1]) {
            flag = 0;
        }

        // exclusive scan of flags (using warp-synchronous version)
        unsigned int offset = flag;
        for (unsigned int d = 1; d < blockDim.x; d <<= 1) {
            unsigned int n = __shfl_up_sync(0xffffffff, offset, d);
            if (tid >= d)
                offset += n;
        }

        // Store unique values only
        if (flag) {
            updated_tracks[offset - 1] = val;
            vecmem::device_atomic_ref<unsigned int> num_updated_tracks(
                (*payload.update_res).n_updated_tracks);
            num_updated_tracks.fetch_add(1);
        }

        // (선택사항) payload.n_updated_unique 같은 걸 추가해서 offset 중 최대값
        // +1 저장하면 unique 개수 알 수 있음
    }

    barrier.blockBarrier();
    */
    /*
    if (threadIdx.x == 0) {
        printf("Number of updated tracks %d \n",
               (*payload.update_res).n_updated_tracks);
        printf("After Sorted updated tracks: ");
        for (int i = 0; i < (*payload.update_res).n_updated_tracks; i++) {
            printf("%d ", updated_tracks[i]);
        }
        printf("\n");
    }
    */
}

}  // namespace traccc::device
