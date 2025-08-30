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

__launch_bounds__(1024) __global__
    void rearrange_tracks(device::rearrange_tracks_payload payload) {

    if (*(payload.terminate) == 1 || *(payload.n_updated_tracks) == 0) {
        return;
    }

    // group (track) index in this block
    const int lane = threadIdx.x % nThreads_per_track;
    const int group = threadIdx.x / nThreads_per_track;
    const bool leader = (lane == 0);

    auto gid = group + blockIdx.x * (blockDim.x / nThreads_per_track);
    const unsigned int n_accepted = *(payload.n_accepted);
    const int N = *(payload.n_updated_tracks);

    int neff_threads = (N + nThreads_per_track - 1) / nThreads_per_track;
    if (neff_threads > nThreads_per_track) {
        neff_threads = nThreads_per_track;
    }

    bool is_valid_thread = true;
    if (lane >= neff_threads || gid >= static_cast<int>(n_accepted)) {
        is_valid_thread = false;
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

    __shared__ int shifted_indices[1024];
    auto& shifted_idx = shifted_indices[group];

    unsigned int tid = std::numeric_limits<unsigned int>::max();

    if (is_valid_thread) {

        tid = sorted_ids[gid];
        const auto rel_sh_ref = rel_shared[tid];
        const auto pval_ref = pvals[tid];

        // initialize once by any lane (all lanes see same reference)
        shifted_idx = static_cast<int>(gid);

        // work partition
        const int stride = (N + neff_threads - 1) / neff_threads;
        const int ini_idx = stride * lane;
        const int fin_idx = min(ini_idx + stride, N);

        if (is_updated[tid]) {

            // ---- group leader: compute base left-shift via binary search over
            // valid (non-updated) slots
            if (gid > 0 && leader) {

                unsigned int left = 0;
                unsigned int right = gid;
                bool first_iteration = true;

                while (right > left) {

                    const bool find_left =
                        find_valid_index(left, 0, gid, sorted_ids, is_updated);
                    if (!find_left)
                        break;

                    const bool find_right =
                        find_valid_index(right, 0, gid, sorted_ids, is_updated);
                    if (!find_right)
                        break;

                    if (first_iteration) {
                        const auto right_idx = sorted_ids[right];
                        const auto rel_sh = rel_shared[right_idx];
                        const auto pval = pvals[right_idx];

                        if (rel_sh < rel_sh_ref ||
                            (rel_sh == rel_sh_ref && pval >= pval_ref)) {
                            left = gid;
                            break;
                        }
                    }
                    first_iteration = false;

                    unsigned int mid = left + (right - left) / 2;
                    const bool find_mid = find_valid_index(
                        mid, left, right - 1, sorted_ids, is_updated);

                    if (find_mid) {
                        const auto mid_idx = sorted_ids[mid];
                        const auto rel_sh = rel_shared[mid_idx];
                        const auto pval = pvals[mid_idx];

                        if (rel_sh < rel_sh_ref ||
                            (rel_sh == rel_sh_ref && pval >= pval_ref)) {
                            left = mid + 1;
                        } else {
                            right = mid;
                        }
                    }
                }

                // BUGFIX: remove duplicate assignment ("delta = delta = ...")
                int delta = static_cast<int>(
                    gid - left - (prefix_sums[gid] - prefix_sums[left]));

                if (!is_updated[sorted_ids[left]]) {
                    delta += 1;
                }

                atomicAdd(&shifted_idx, -delta);
            }

            // ---- all lanes: single-pass over [ini_idx, fin_idx) to (a) count
            // left-updates, (b) find offset
            int local_delta = 0;
            int local_offset = -1;

            for (int i = ini_idx; i < fin_idx; ++i) {
                const auto id = updated_tracks[i];

                // how many updated tracks originally to the left of gid
                if (inverted_ids[id] < static_cast<unsigned int>(gid)) {
                    local_delta -= 1;
                }

                // find the position of my tid in updated_tracks
                if (local_offset < 0 && id == tid) {
                    local_offset =
                        i;  // if i == 0, adding 0 is a no-op (original logic)
                }
            }

            if (local_delta != 0) {
                atomicAdd(&shifted_idx, local_delta);
            }
            if (local_offset > 0) {
                atomicAdd(&shifted_idx, local_offset);
            }

        } else {
            // tid is NOT updated: count how many updated tracks should move to
            // the right of me
            int local_delta = 0;

            for (int i = ini_idx; i < fin_idx; ++i) {
                const auto id = updated_tracks[i];
                if (inverted_ids[id] > static_cast<unsigned int>(gid)) {
                    const auto rel_sh = rel_shared[id];
                    const auto pval = pvals[id];

                    if (rel_sh < rel_sh_ref ||
                        (rel_sh == rel_sh_ref && pval > pval_ref)) {
                        local_delta += 1;
                    }
                }
            }

            if (local_delta != 0) {
                atomicAdd(&shifted_idx, local_delta);
            }
        }
    }

    __syncthreads();

    if (is_valid_thread && leader) {
        temp_sorted_ids.at(shifted_idx) = tid;
    }
}

}  // namespace traccc::cuda::kernels
