/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s)
#include <cmath>

// Project include(s).
#include "traccc/seeding/seed_selecting_helper.hpp"

namespace traccc::device {

// Finding minimum element algorithm
template <typename Comparator>
TRACCC_HOST_DEVICE static int min_elem(triplet* arr, const int& begin_idx,
                                       const int& end_idx, Comparator comp) {
    int min_i = begin_idx;
    auto next = begin_idx;
    while (next < end_idx - 1) {
        ++next;
        if (comp(arr[min_i], arr[next])) {
            min_i = next;
        }
    }
    return min_i;
}

// Sorting algorithm for sorting seeds in the local memory
template <typename Comparator>
TRACCC_HOST_DEVICE static void insertionSort(triplet* arr, const int& begin_idx,
                                             const int& n, Comparator comp) {
    int j = 0;
    triplet key = arr[begin_idx];
    for (int i = 0; i < n; ++i) {
        key = arr[begin_idx + i];
        j = i - 1;
        while (j >= 0 && !comp(arr[begin_idx + j], key)) {
            arr[begin_idx + j + 1] = arr[begin_idx + j];
            j = j - 1;
        }
        arr[begin_idx + j + 1] = key;
    }
}

// Select seeds kernel
TRACCC_HOST_DEVICE
void select_seeds(
    const std::size_t globalIndex, const seedfilter_config& filter_config,
    const spacepoint_container_types::const_view& spacepoints_view,
    const sp_grid_const_view& internal_sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& dc_ps_view,
    const device::doublet_counter_container_types::const_view&
        doublet_counter_container,
    const triplet_container_view& triplet_view,
    vecmem::data::vector_view<seed> seed_view) {

    // Check if anything needs to be done.
    const vecmem::device_vector<const prefix_sum_element_t> dc_prefix_sum(
        dc_ps_view);
    if (globalIndex >= dc_prefix_sum.size()) {
        return;
    }

    // Set up the device containers
    const spacepoint_container_types::const_device spacepoints_device(
        spacepoints_view);
    const const_sp_grid_device internal_sp_device(internal_sp_view);

    const prefix_sum_element_t ps_idx = dc_prefix_sum[globalIndex];
    const auto bin_idx = ps_idx.first;

    device::doublet_counter_container_types::const_device
        doublet_counter_device(doublet_counter_container);
    device_triplet_container triplet_device(triplet_view);
    device_seed_collection seed_device(seed_view);

    // Header of triplet: number of triplets per bin
    // Item of triplet: triplet objects per bin
    const auto num_triplets_per_bin =
        triplet_device.get_headers().at(bin_idx).n_triplets;
    const auto triplets_per_bin = triplet_device.get_items().at(bin_idx);

    // Current work item = middle spacepoint
    const auto spM_loc =
        doublet_counter_device.get_items().at(bin_idx)[ps_idx.second].m_spM;
    const auto spM = internal_sp_device.bin(bin_idx)[spM_loc.sp_idx];

    // Number of triplets added for this spM
    unsigned int n_triplets_per_spM = 0;

    static const std::size_t MAX_TRIPLETS_SPM = 5;
    assert(MAX_TRIPLETS_SPM >= filter_config.max_triplets_per_spM);
    // Triplets added for this spM
    triplet triplets_per_spM[MAX_TRIPLETS_SPM];

    // iterate over the triplets in the bin
    for (unsigned int i = 0; i < num_triplets_per_bin; ++i) {
        auto aTriplet = triplets_per_bin[i];

        // consider only the triplets with the same middle spacepoint
        if (!(spM_loc == aTriplet.sp2)) {
            continue;
        }

        // spacepoints bottom and top for this triplet
        const auto& spB_loc = aTriplet.sp1;
        const auto& spT_loc = aTriplet.sp3;
        const auto& spB =
            internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        const auto& spT =
            internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

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

            int min_index = min_elem(triplets_per_spM, 0,
                                     filter_config.max_triplets_per_spM,
                                     [&](triplet& lhs, triplet& rhs) {
                                         return lhs.weight > rhs.weight;
                                     });

            auto& min_weight = triplets_per_spM[min_index].weight;

            if (aTriplet.weight > min_weight) {
                triplets_per_spM[min_index] = aTriplet;
            }
        }

        // if the number of good triplets is below the threshold, add
        // the current triplet to the array
        else if (n_triplets_per_spM < filter_config.max_triplets_per_spM) {
            triplets_per_spM[n_triplets_per_spM] = aTriplet;
            n_triplets_per_spM++;
        }
    }

    // sort the triplets per spM
    insertionSort(
        triplets_per_spM, 0, n_triplets_per_spM,
        [&](triplet& lhs, triplet& rhs) {
            if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
            } else {

                scalar seed1_sum = 0;
                scalar seed2_sum = 0;

                auto& ispB1 =
                    internal_sp_device.bin(lhs.sp1.bin_idx)[lhs.sp1.sp_idx];
                auto& ispT1 =
                    internal_sp_device.bin(lhs.sp3.bin_idx)[lhs.sp3.sp_idx];
                auto& ispB2 =
                    internal_sp_device.bin(rhs.sp1.bin_idx)[rhs.sp1.sp_idx];
                auto& ispT2 =
                    internal_sp_device.bin(rhs.sp3.bin_idx)[rhs.sp3.sp_idx];

                const auto& spB1 = spacepoints_device.at(ispB1.m_link);
                const auto& spT1 = spacepoints_device.at(ispT1.m_link);
                const auto& spB2 = spacepoints_device.at(ispB2.m_link);
                const auto& spT2 = spacepoints_device.at(ispT2.m_link);

                constexpr scalar exp = 2;
                seed1_sum += std::pow(spB1.y(), exp) + std::pow(spB1.z(), exp);
                seed1_sum += std::pow(spT1.y(), exp) + std::pow(spT1.z(), exp);
                seed2_sum += std::pow(spB2.y(), exp) + std::pow(spB2.z(), exp);
                seed2_sum += std::pow(spT2.y(), exp) + std::pow(spT2.z(), exp);

                return seed1_sum > seed2_sum;
            }
        });

    // the number of good seed per compatible middle spacepoint
    unsigned int n_seeds_per_spM = 0;

    // iterate over the good triplets for final selection of seeds
    for (unsigned int i = 0; i < n_triplets_per_spM; ++i) {
        auto& aTriplet = triplets_per_spM[i];
        auto& spB_loc = aTriplet.sp1;
        auto& spT_loc = aTriplet.sp3;
        auto& spB = internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        auto& spT = internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

        // if the number of seeds reaches the threshold, break
        if (n_seeds_per_spM >= filter_config.maxSeedsPerSpM + 1) {
            break;
        }

        seed aSeed({spB.m_link, spM.m_link, spT.m_link, aTriplet.weight,
                    aTriplet.z_vertex});

        // check if it is a good triplet
        if (seed_selecting_helper::cut_per_middle_sp(
                filter_config, spacepoints_device, aSeed, aTriplet.weight) ||
            n_seeds_per_spM == 0) {

            n_seeds_per_spM++;

            seed_device.push_back(aSeed);
        }
    }
}

}  // namespace traccc::device
