/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Thrust include(s).
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace traccc::device {

// Track comparator to sort the track ids
struct track_comparator {
    const traccc::scalar* rel_shared;
    const traccc::scalar* pvals;

    TRACCC_HOST_DEVICE track_comparator(const traccc::scalar* rel_shared_,
                                        const traccc::scalar* pvals_)
        : rel_shared(rel_shared_), pvals(pvals_) {}

    TRACCC_HOST_DEVICE bool operator()(unsigned int a, unsigned int b) const {
        if (rel_shared[a] != rel_shared[b]) {
            return rel_shared[a] < rel_shared[b];
        }
        return pvals[a] > pvals[b];
    }
};

TRACCC_HOST_DEVICE inline void check_sortedness(
    const global_index_t globalIndex, const check_sortedness_payload& payload) {

    if (globalIndex >= payload.n_updated_tracks) {
        return;
    }

    vecmem::device_vector<const unsigned int> sorted_ids(
        payload.sorted_ids_view);
    vecmem::device_vector<const traccc::scalar> rel_shared(
        payload.rel_shared_view);
    vecmem::device_vector<const traccc::scalar> pvals(payload.pvals_view);
    vecmem::device_vector<const unsigned int> updated_tracks(
        payload.updated_tracks_view);

    const auto tid = updated_tracks[globalIndex];

    const auto it =
        thrust::find(thrust::seq, sorted_ids.begin(), sorted_ids.end(), tid);
    /*
    const auto it2 = thrust::lower_bound(thrust::seq, sorted_ids.begin(),
                                         sorted_ids.end(), tid, trk_comp);

    printf("tid %d *it %d \n", tid, *it2);

    if (tid != *it2) {
        *payload.do_sort = true;
    }
    */
    /*
    track_comparator trk_comp(rel_shared.begin(), pvals.begin());

    const auto it = thrust::lower_bound(thrust::seq, sorted_ids.begin(),
                                        sorted_ids.end(), tid, trk_comp);

    printf("tid %d *it %d \n", tid, *it);

    if (tid != *it) {
        *payload.do_sort = true;
    }
    */

    /*
    const auto it =
        thrust::find(thrust::seq, sorted_ids.begin(), sorted_ids.end(), tid);

    if (it != sorted_ids.begin()) {

        const unsigned int it_idx =
            static_cast<unsigned int>(thrust::distance(sorted_ids.begin(), it));

        const auto prev_tid = updated_tracks[it_idx - 1];

        if (rel_shared[prev_tid] > rel_shared[tid]) {
            *payload.do_sort = true;
        } else if (rel_shared[prev_tid] == rel_shared[tid]) {
            if (pvals[prev_tid] < pvals[tid]) {
                *payload.do_sort = true;
            }
        }
    }
    */
}

}  // namespace traccc::device
