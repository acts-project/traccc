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
TRACCC_HOST_DEVICE void insertionSort(triplet* arr, const std::size_t begin_idx,
                                      const std::size_t n, Comparator comp) {
    int j = 0;
    triplet key = arr[begin_idx];
    for (std::size_t i = 0; i < n; ++i) {
        key = arr[begin_idx + i];
        j = i - 1;
        while (j >= 0 && !comp(arr[begin_idx + j], key)) {
            arr[begin_idx + j + 1] = arr[begin_idx + j];
            j = j - 1;
        }
        arr[begin_idx + j + 1] = key;
    }
}
}  // namespace details

// Select seeds kernel
TRACCC_HOST_DEVICE
void select_seeds(
    const std::size_t globalIndex, const seedfilter_config& filter_config,
    const spacepoint_container_types::const_view& spacepoints_view,
    const sp_grid_const_view& internal_sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& dc_ps_view,
    const device::doublet_counter_container_types::const_view&
        doublet_counter_container,
    const triplet_container_view& triplet_view, triplet* data,
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
    const std::size_t bin_idx = ps_idx.first;

    device::doublet_counter_container_types::const_device
        doublet_counter_device(doublet_counter_container);
    device_triplet_container triplet_device(triplet_view);
    device_seed_collection seed_device(seed_view);

    // Header of triplet: number of triplets per bin
    // Item of triplet: triplet objects per bin
    const unsigned int num_triplets_per_bin =
        triplet_device.get_headers().at(bin_idx).n_triplets;
    const vecmem::device_vector<triplet> triplets_per_bin =
        triplet_device.get_items().at(bin_idx);

    // Current work item = middle spacepoint
    const sp_location spM_loc =
        doublet_counter_device.get_items().at(bin_idx)[ps_idx.second].m_spM;
    const internal_spacepoint<spacepoint> spM =
        internal_sp_device.bin(bin_idx)[spM_loc.sp_idx];

    // Number of triplets added for this spM
    unsigned int n_triplets_per_spM = 0;

    // iterate over the triplets in the bin
    for (unsigned int i = 0; i < num_triplets_per_bin; ++i) {
        triplet aTriplet = triplets_per_bin[i];

        // consider only the triplets with the same middle spacepoint
        if (spM_loc != aTriplet.sp2) {
            continue;
        }

        // spacepoints bottom and top for this triplet
        const sp_location& spB_loc = aTriplet.sp1;
        const sp_location& spT_loc = aTriplet.sp3;
        const internal_spacepoint<spacepoint> spB =
            internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        const internal_spacepoint<spacepoint> spT =
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

            const int min_index =
                details::min_elem(data, 0, filter_config.max_triplets_per_spM,
                                  [](const triplet lhs, const triplet rhs) {
                                      return lhs.weight > rhs.weight;
                                  });

            const scalar& min_weight = data[min_index].weight;

            if (aTriplet.weight > min_weight) {
                data[min_index] = aTriplet;
            }
        }

        // if the number of good triplets is below the threshold, add
        // the current triplet to the array
        else if (n_triplets_per_spM < filter_config.max_triplets_per_spM) {
            data[n_triplets_per_spM] = aTriplet;
            n_triplets_per_spM++;
        }
    }

    // sort the triplets per spM
    details::insertionSort(
        data, 0, n_triplets_per_spM, [&](triplet& lhs, triplet& rhs) {
            if (lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
            } else {

                scalar seed1_sum = 0;
                scalar seed2_sum = 0;

                const internal_spacepoint<spacepoint>& ispB1 =
                    internal_sp_device.bin(lhs.sp1.bin_idx)[lhs.sp1.sp_idx];
                const internal_spacepoint<spacepoint>& ispT1 =
                    internal_sp_device.bin(lhs.sp3.bin_idx)[lhs.sp3.sp_idx];
                const internal_spacepoint<spacepoint>& ispB2 =
                    internal_sp_device.bin(rhs.sp1.bin_idx)[rhs.sp1.sp_idx];
                const internal_spacepoint<spacepoint>& ispT2 =
                    internal_sp_device.bin(rhs.sp3.bin_idx)[rhs.sp3.sp_idx];

                const spacepoint& spB1 = spacepoints_device.at(ispB1.m_link);
                const spacepoint& spT1 = spacepoints_device.at(ispT1.m_link);
                const spacepoint& spB2 = spacepoints_device.at(ispB2.m_link);
                const spacepoint& spT2 = spacepoints_device.at(ispT2.m_link);

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
        const triplet& aTriplet = data[i];
        const sp_location& spB_loc = aTriplet.sp1;
        const sp_location& spT_loc = aTriplet.sp3;
        const internal_spacepoint<spacepoint>& spB =
            internal_sp_device.bin(spB_loc.bin_idx)[spB_loc.sp_idx];
        const internal_spacepoint<spacepoint>& spT =
            internal_sp_device.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

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
