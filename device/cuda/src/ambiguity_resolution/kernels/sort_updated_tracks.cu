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

__launch_bounds__(512) __global__
    void sort_updated_tracks(device::sort_updated_tracks_payload payload) {

    const unsigned int n_updated = *(payload.n_updated_tracks);

    if (*(payload.terminate) == 1 || n_updated == 0 || n_updated == 1) {
        return;
    }

    // Shared: track id + keys cached once (no global reads inside compare
    // loops)
    __shared__ unsigned int sh_trk[512];
    __shared__ traccc::scalar sh_rel[512];
    __shared__ traccc::scalar sh_pval[512];

    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<unsigned int> updated_tracks(
        payload.updated_tracks_view);

    const unsigned int tid = threadIdx.x;

    // Padding the number of tracks to the power of 2
    const unsigned int N = 1 << (32 - __clz(n_updated - 1));

    // Sentinel keys to push to the end
    const unsigned int TRK_SENT = std::numeric_limits<unsigned int>::max();
    const traccc::scalar REL_INF =
        std::numeric_limits<traccc::scalar>::infinity();
    const traccc::scalar PVAL_MIN = traccc::scalar(0);

    // Load once to shared (coalesced on updated_tracks)
    if (tid < n_updated) {
        unsigned int trk = updated_tracks[tid];
        sh_trk[tid] = trk;
        sh_rel[tid] = rel_shared[trk];
        sh_pval[tid] = pvals[trk];
    } else {
        sh_trk[tid] = TRK_SENT;
        sh_rel[tid] =
            REL_INF;  // bigger rel → goes to the end for ascending rel
        sh_pval[tid] = PVAL_MIN;  // tie-breaker doesn't matter once rel=INF
    }

    // For any threads beyond N, still need to participate in barriers,
    // but give them sentinel content so they don't affect ordering.
    if (tid >= N) {
        sh_trk[tid] = TRK_SENT;
        sh_rel[tid] = REL_INF;
        sh_pval[tid] = PVAL_MIN;
    }

    __syncthreads();

    // Bitonic sort over shared data only
    for (unsigned int k = 2; k <= N; k <<= 1) {

        const bool ascending = ((tid & k) == 0);

        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            const unsigned int ixj = tid ^ j;

            if (ixj > tid && ixj < N && tid < N) {
                // Load to registers
                unsigned int trk_i = sh_trk[tid];
                unsigned int trk_j = sh_trk[ixj];
                traccc::scalar rel_i = sh_rel[tid];
                traccc::scalar rel_j = sh_rel[ixj];
                traccc::scalar pval_i = sh_pval[tid];
                traccc::scalar pval_j = sh_pval[ixj];

                // Compare: ascending by rel, and for equal rel, descending by
                // pval
                const bool greater =
                    (rel_i > rel_j) || ((rel_i == rel_j) && (pval_i < pval_j));

                const bool should_swap = (greater == ascending);
                if (should_swap) {
                    // swap triad
                    sh_trk[tid] = trk_j;
                    sh_trk[ixj] = trk_i;
                    sh_rel[tid] = rel_j;
                    sh_rel[ixj] = rel_i;
                    sh_pval[tid] = pval_j;
                    sh_pval[ixj] = pval_i;
                }
            }
            __syncthreads();
        }
    }

    if (tid < n_updated) {
        updated_tracks[tid] = sh_trk[tid];
    }
}

}  // namespace traccc::cuda::kernels
