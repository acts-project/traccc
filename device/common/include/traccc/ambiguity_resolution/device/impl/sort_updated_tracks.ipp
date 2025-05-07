/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

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

    // Bitonic sort (ascending on rel_shared, descending on pvals)
    for (unsigned int k = 2; k <= N; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;
            if (ixj > tid && ixj < N && tid < N) {
                unsigned int a = shared_mem_tracks[tid];
                unsigned int b = shared_mem_tracks[ixj];

                bool ascending = ((tid & k) == 0);

                traccc::scalar rel_a = rel_shared[a];
                traccc::scalar rel_b = rel_shared[b];
                traccc::scalar pv_a = pvals[a];
                traccc::scalar pv_b = pvals[b];

                bool swap = false;

                if (rel_a != rel_b) {
                    swap = ascending ? (rel_a > rel_b) : (rel_a < rel_b);
                } else {
                    swap = ascending ? (pv_a < pv_b) : (pv_a > pv_b);
                }

                if (swap) {
                    shared_mem_tracks[tid] = b;
                    shared_mem_tracks[ixj] = a;
                }
            }
            barrier.blockBarrier();
        }
    }

    // Write back
    if (tid < N) {
        updated_tracks[tid] = shared_mem_tracks[tid];
    }
}

}  // namespace traccc::device
