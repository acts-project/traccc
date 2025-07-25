/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// Local include(s).
#include "../../utils/global_index.hpp"
#include "rearrange_tracks.cuh"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>

namespace traccc::cuda::kernels {

TRACCC_DEVICE inline bool find_valid_index(
    unsigned int& idx, const int lower_bound, const int upper_bound,
    const vecmem::device_vector<const unsigned int>& sorted_ids,
    const vecmem::device_vector<const int>& is_updated) {

    const auto initial_idx = idx;

    for (int i = initial_idx; i <= upper_bound; i++) {
        if (!is_updated[sorted_ids[i]]) {
            idx = i;
            return true;
        }
    }

    for (int i = initial_idx - 1; i >= lower_bound; i--) {
        if (!is_updated[sorted_ids[i]]) {
            idx = i;
            return true;
        }
    }

    return false;
}

__global__ void rearrange_tracks(device::rearrange_tracks_payload payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    auto gid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int n_accepted = *(payload.n_accepted);

    if (gid >= n_accepted) {
        return;
    }

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
    vecmem::device_vector<const int> prefix_sums(payload.prefix_sums_view);
    vecmem::device_vector<unsigned int> temp_sorted_ids(
        payload.temp_sorted_ids_view);

    const auto tid = sorted_ids[gid];
    auto rel_sh_ref = rel_shared[tid];
    auto pval_ref = pvals[tid];
    int shifted_idx = static_cast<int>(gid);
    auto N = *(payload.n_updated_tracks);

    if (is_updated[tid]) {

        if (gid > 0) {

            unsigned int left = 0;
            unsigned int right = gid;

            bool first_iteration = true;
            while (right > left) {

                const bool find_left =
                    find_valid_index(left, 0, gid, sorted_ids, is_updated);

                if (!find_left) {
                    break;
                }

                const bool find_right =
                    find_valid_index(right, 0, gid, sorted_ids, is_updated);

                if (!find_right) {
                    break;
                }

                if (first_iteration) {
                    auto rel_sh = rel_shared[sorted_ids[right]];
                    auto pval = pvals[sorted_ids[right]];

                    if (rel_sh < rel_sh_ref ||
                        (rel_sh == rel_sh_ref && pval >= pval_ref)) {
                        left = gid;
                        break;
                    }
                }

                first_iteration = false;

                unsigned int mid = left + (right - left) / 2;

                const bool find_mid = find_valid_index(mid, left, right - 1,
                                                       sorted_ids, is_updated);

                if (find_mid) {

                    auto rel_sh = rel_shared[sorted_ids[mid]];
                    auto pval = pvals[sorted_ids[mid]];

                    if (rel_sh < rel_sh_ref ||
                        (rel_sh == rel_sh_ref && pval >= pval_ref)) {

                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
            }

            int delta = 0;

            if (is_updated[sorted_ids[left]]) {
                delta = gid - left - (prefix_sums[gid] - prefix_sums[left]);
            } else {
                delta = gid - left - (prefix_sums[gid] - prefix_sums[left] - 1);
            }

            shifted_idx -= delta;
        }

        for (int i = 0; i < N; i++) {

            auto id = updated_tracks[i];

            if (inverted_ids[id] < gid) {
                shifted_idx--;
            }
        }

        int offset = 0;
        for (int i = 0; i < N; i++) {
            if (updated_tracks[i] == tid) {
                offset = i;
                break;
            }
        }
        shifted_idx += offset;

    } else {
        for (int i = 0; i < N; i++) {

            auto id = updated_tracks[i];
            auto rel_sh = rel_shared[id];
            auto pval = pvals[id];

            if (inverted_ids[id] > gid) {
                if (rel_sh < rel_sh_ref) {
                    shifted_idx++;
                } else if (rel_sh == rel_sh_ref && pval > pval_ref) {
                    shifted_idx++;
                }
            }
        }
    }

    temp_sorted_ids.at(shifted_idx) = tid;
}

}  // namespace traccc::cuda::kernels
