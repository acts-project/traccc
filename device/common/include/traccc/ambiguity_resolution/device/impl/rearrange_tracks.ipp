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

// Track comparator to sort the track ids
struct track_comparator {
    const vecmem::device_vector<const traccc::scalar>& rel_shared;
    const vecmem::device_vector<const traccc::scalar>& pvals;

    TRACCC_HOST_DEVICE track_comparator(
        const vecmem::device_vector<const traccc::scalar>& rel_shared_,
        const vecmem::device_vector<const traccc::scalar>& pvals_)
        : rel_shared(rel_shared_), pvals(pvals_) {}

    TRACCC_HOST_DEVICE bool operator()(unsigned int a, unsigned int b) const {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    }
};

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

    // Found the ids in the updated tracks
    const auto it_end =
        updated_tracks.begin() + (*payload.update_res).n_updated_tracks;    
    auto it = thrust::find(thrust::seq, updated_tracks.begin(), it_end, tid);
    if (it != it_end) {
        track_comparator trk_comp(rel_shared, pvals);
        auto id = *it;
        // index of sorted_ids
        auto sid = inverted_ids[id];
        auto it2 = thrust::lower_bound(thrust::seq, sorted_ids.begin(),
                                       sorted_ids.begin() + sid, sid, trk_comp);

        auto shifted_idx = it2 - sorted_ids.begin();
    }

    for (int i = 0; i < (*payload.update_res).n_updated_tracks; i++) {

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

    temp_sorted_ids[shifted_idx] = tid;
}

}  // namespace traccc::device
