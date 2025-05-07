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
TRACCC_DEVICE inline void sort_tracks(const global_index_t globalIndex,
                                      const barrier_t& barrier,
                                      const sort_tracks_payload& payload) {

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<const unsigned int> updated_tracks(
        payload.updated_tracks_view);
    vecmem::device_vector<unsigned int> sorted_ids(payload.sorted_ids_view);

    const auto n_iter = (*payload.update_res).n_accepted / blockDim.x + 1;

    for (unsigned int i = 0; i < (*payload.update_res).n_updated_tracks; i++) {

        if (threadIdx.x == 0) {
            printf("test2 \n");
        }

        unsigned int tid = updated_tracks[i];
        unsigned int sid;

        for (unsigned int j = 0; j < n_iter; j++) {

            if (threadIdx.x < (*payload.update_res).n_accepted) {
                if (sorted_ids[threadIdx.x] == tid) {
                    sid = threadIdx.x;
                }
            }
        }

        if (threadIdx.x == 0) {
            printf("test3 \n");
        }

        barrier.blockBarrier();


        if (threadIdx.x == 0) {
            printf("test4 \n");
        }

        // Do the bitonic sort
        int N = min(blockDim.x, sid);
        int id = sid - N + threadIdx.x;


        if (threadIdx.x == 0) {
            printf("test5 \n");
        }

        if (threadIdx.x == 0) {
            printf("%d %d %d %d \n", N, id, i,
                   (*payload.update_res).n_updated_tracks);
        }

        if (id < (*payload.update_res).n_accepted) {
            for (int k = 2; k <= N; k *= 2) {
                for (int j = k / 2; j > 0; j /= 2) {

                    if (threadIdx.x == 0) {
                        printf("%d %d \n", k, j);
                    }

                    int ixj = id ^ j;

                    if (ixj > id && ixj < N) {
                        unsigned int a = sorted_ids[id];
                        unsigned int b = sorted_ids[ixj];

                        float rel_a = rel_shared[a];
                        float rel_b = rel_shared[b];
                        float pval_a = pvals[a];
                        float pval_b = pvals[b];

                        bool swap = false;
                        if ((id & k) == 0) {
                            // ascending
                            swap = (rel_a > rel_b) ||
                                   (rel_a == rel_b && pval_a < pval_b);
                        } else {
                            // descending
                            swap = (rel_a < rel_b) ||
                                   (rel_a == rel_b && pval_a > pval_b);
                        }

                        if (swap) {
                            sorted_ids[id] = b;
                            sorted_ids[ixj] = a;
                        }
                    }

                    barrier.blockBarrier();
                }
            }
        }
        if (threadIdx.x == 0) {
            printf("test1 \n");
        }
    }

    printf("Done \n");
}

}  // namespace traccc::device
