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
    const sort_updated_tracks_payload& payload,
    unsigned int* shared_mem_tracks) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    const unsigned int tid = globalIndex;
    const unsigned int N = *(payload.n_updated_tracks);

    // Load to shared memory
    if (tid < N) {
        shared_mem_tracks[tid] = updated_tracks[tid];
    }

    barrier.blockBarrier();

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
        barrier.blockBarrier();
    }

    if (tid < N) {
        updated_tracks[tid] = shared_mem_tracks[tid];
    }
}

}  // namespace traccc::device
