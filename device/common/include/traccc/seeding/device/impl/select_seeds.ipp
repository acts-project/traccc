/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/detail/triplet_sorter.hpp"
#include "traccc/seeding/seed_selecting_helper.hpp"

// System include(s)
#include <cassert>
#include <cmath>

namespace traccc::device {

namespace details {
// Finding minimum element algorithm
template <typename Comparator>
TRACCC_HOST_DEVICE std::size_t min_elem(const triplet* arr,
                                        const std::size_t begin_idx,
                                        const std::size_t end_idx,
                                        Comparator comp) {
    assert(begin_idx <= end_idx);
    std::size_t min_i = begin_idx;
    std::size_t next = begin_idx;
    for (std::size_t i = begin_idx + 1; i < end_idx; ++i) {
        ++next;
        if (comp(arr[min_i], arr[next])) {
            min_i = next;
        }
    }
    return min_i;
}

// Sorting algorithm for sorting seeds in the local memory
template <typename Comparator>
TRACCC_HOST_DEVICE void insertionSort(triplet* arr,
                                      const unsigned int begin_idx,
                                      const unsigned int n, Comparator comp) {
    int j = 0;
    triplet key = arr[begin_idx];
    for (unsigned int i = 0; i < n; ++i) {
        key = arr[begin_idx + i];
        j = static_cast<int>(i) - 1;
        while (j >= 0 &&
               !comp(arr[begin_idx + static_cast<unsigned int>(j)], key)) {
            arr[begin_idx + static_cast<unsigned int>(j + 1)] =
                arr[begin_idx + static_cast<unsigned int>(j)];
            j = j - 1;
        }
        arr[begin_idx + static_cast<unsigned int>(j + 1)] = key;
    }
}
}  // namespace details

// Select seeds kernel
TRACCC_HOST_DEVICE
inline void select_seeds(
    const global_index_t globalIndex, const seedfilter_config& filter_config,
    const edm::spacepoint_collection::const_view& spacepoints_view,
    const traccc::details::spacepoint_grid_types::const_view& sp_view,
    const triplet_counter_spM_collection_types::const_view& spM_tc_view,
    const triplet_counter_collection_types::const_view& tc_view,
    const device_triplet_collection_types::const_view& triplet_view,
    triplet* data, edm::seed_collection::view seed_view) {

    // Check if anything needs to be done.
    const triplet_counter_spM_collection_types::const_device triplet_counts_spM(
        spM_tc_view);
    if (globalIndex >= triplet_counts_spM.size()) {
        return;
    }

    // Set up the device containers
    const triplet_counter_collection_types::const_device triplet_counts(
        tc_view);
    const edm::spacepoint_collection::const_device spacepoints{
        spacepoints_view};
    const traccc::details::spacepoint_grid_types::const_device sp_device(
        sp_view);

    device_triplet_collection_types::const_device triplets(triplet_view);
    edm::seed_collection::device seeds_device(seed_view);

    // Current work item = middle spacepoint
    const triplet_counter_spM spM_counter = triplet_counts_spM.at(globalIndex);
    const sp_location spM_loc = spM_counter.spM;
    const edm::spacepoint_collection::const_device::const_proxy_type spM =
        spacepoints.at(sp_device.bin(spM_loc.bin_idx)[spM_loc.sp_idx]);

    // Number of triplets added for this spM
    unsigned int n_triplets_per_spM = 0;

    const unsigned int end_triplets_spM =
        spM_counter.posTriplets + spM_counter.m_nTriplets;
    // iterate over the triplets in the bin
    for (unsigned int i = spM_counter.posTriplets; i < end_triplets_spM; ++i) {
        device_triplet aTriplet = triplets[i];

        // spacepoints bottom and top for this triplet
        const sp_location spB_loc =
            triplet_counts.at(static_cast<unsigned int>(aTriplet.counter_link))
                .spB;
        const sp_location spT_loc = aTriplet.spT;
        const edm::spacepoint_collection::const_device::const_proxy_type spB =
            spacepoints.at(sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx]);
        const edm::spacepoint_collection::const_device::const_proxy_type spT =
            spacepoints.at(sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx]);

        // update weight of triplet
        seed_selecting_helper::seed_weight(filter_config, spM, spB, spT,
                                           aTriplet.weight);

        // check if it is a good triplet
        if (!seed_selecting_helper::single_seed_cut(filter_config, spM, spB,
                                                    spT, aTriplet.weight)) {
            continue;
        }

        // if the number of good triplets is larger than the threshold,
        // the triplet with the lowest weight is removed
        if (n_triplets_per_spM >= filter_config.max_triplets_per_spM) {

            const std::size_t min_index =
                details::min_elem(data, 0, filter_config.max_triplets_per_spM,
                                  [](const triplet lhs, const triplet rhs) {
                                      return lhs.weight > rhs.weight;
                                  });

            const scalar& min_weight = data[min_index].weight;

            if (aTriplet.weight > min_weight) {
                data[min_index] = {spB_loc,         spM_loc,
                                   spT_loc,         aTriplet.curvature,
                                   aTriplet.weight, aTriplet.z_vertex};
            }
        }

        // if the number of good triplets is below the threshold, add
        // the current triplet to the array
        else if (n_triplets_per_spM < filter_config.max_triplets_per_spM) {
            data[n_triplets_per_spM] = {spB_loc,         spM_loc,
                                        spT_loc,         aTriplet.curvature,
                                        aTriplet.weight, aTriplet.z_vertex};
            n_triplets_per_spM++;
        }
    }

    // sort the triplets per spM
    details::insertionSort(
        data, 0, n_triplets_per_spM,
        traccc::details::triplet_sorter{spacepoints, sp_device});

    // the number of good seed per compatible middle spacepoint
    unsigned int n_seeds_per_spM = 0;

    // iterate over the good triplets for final selection of seeds
    for (unsigned int i = 0; i < n_triplets_per_spM; ++i) {
        const triplet& aTriplet = data[i];
        const sp_location& spB_loc = aTriplet.sp1;
        const sp_location& spT_loc = aTriplet.sp3;

        // if the number of seeds reaches the threshold, break
        if (n_seeds_per_spM >= filter_config.maxSeedsPerSpM + 1) {
            break;
        }

        // check if it is a good triplet
        if (seed_selecting_helper::cut_per_middle_sp(filter_config, spacepoints,
                                                     sp_device, aTriplet) ||
            n_seeds_per_spM == 0) {

            n_seeds_per_spM++;

            const edm::seed_collection::device::size_type iseed =
                seeds_device.push_back_default();
            edm::seed_collection::device::proxy_type seed =
                seeds_device.at(iseed);
            seed.bottom_index() =
                sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
            seed.middle_index() =
                sp_device.bin(spM_loc.bin_idx)[spM_loc.sp_idx];
            seed.top_index() = sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];
        }
    }
}

}  // namespace traccc::device
