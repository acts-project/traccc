/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Traccc include(s).
#include "traccc/device/concepts/barrier.hpp"

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_DEVICE inline void rearrange_tracks(
    const global_index_t globalIndex, const barrier_t& barrier,
    const rearrange_tracks_payload& payload) {

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const unsigned int> inverted_ids(
        payload.inverted_ids_view);
    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<const unsigned int> updated_tracks(
        payload.updated_tracks_view);
    vecmem::device_vector<const int> is_updated(payload.is_updated_view);
    vecmem::device_vector<unsigned int> temp_sorted_ids(
        payload.temp_sorted_ids_view);

    const unsigned int n_accepted = (*payload.update_res).n_accepted;

    if (globalIndex >= n_accepted) {
        return;
    }

    const auto tid = sorted_ids[globalIndex];
    auto rel_sh_ref = rel_shared[tid];
    auto pval_ref = pvals[tid];
    auto shifted_idx = globalIndex;

    auto N = (*payload.update_res).n_updated_tracks;
    // Found the ids in the updated tracks
    /*
    const auto it_end =
        updated_tracks.begin() + (*payload.update_res).n_updated_tracks;

    auto it = thrust::find(thrust::seq, updated_tracks.begin(), it_end, tid);

    if (it != it_end) {
    */
    if (is_updated[tid]) {
        // index of sorted_ids
        auto sid = inverted_ids[tid];

        //int k = 0;

        if (sid > 0) {

            // Use binary search here
            for (int i = sid - 1; i >= 0; i--) {

                auto tid2 = sorted_ids[i];
                /*
                auto it2 = thrust::find(thrust::seq, updated_tracks.begin(),
                                        it_end, tid2);

                if (it2 == it_end) {
                */
                if (!is_updated[tid2]) {
                    auto rel_sh = rel_shared[tid2];
                    auto pval = pvals[tid2];

                    if (rel_sh > rel_sh_ref ||
                        (rel_sh == rel_sh_ref && pval < pval_ref)) {

                        shifted_idx--;
                    } else {
                        //k = i;
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < N; i++) {

            auto id = updated_tracks[i];
            auto rel_sh = rel_shared[id];
            auto pval = pvals[id];

            if (inverted_ids[id] < globalIndex) {
                shifted_idx--;
            }
        }

        //printf("%d %d \n", sid - k, N);

        auto it = thrust::find(thrust::seq, updated_tracks.begin(),
                               updated_tracks.begin() + N, tid);

        shifted_idx += it - updated_tracks.begin();
    } else {
        for (int i = 0; i < N; i++) {

            auto id = updated_tracks[i];
            auto rel_sh = rel_shared[id];
            auto pval = pvals[id];

            if (inverted_ids[id] > globalIndex) {
                if (rel_sh < rel_sh_ref) {
                    shifted_idx++;
                } else if (rel_sh == rel_sh_ref && pval > pval_ref) {
                    shifted_idx++;
                }
            }
        }
    }

    temp_sorted_ids[shifted_idx] = tid;
}

}  // namespace traccc::device
