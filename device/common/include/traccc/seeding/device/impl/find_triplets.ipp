/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/triplet_finding_helper.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
void find_triplets(
    const std::size_t globalIndex, const seedfinder_config& config,
    const seedfilter_config& filter_config, const sp_grid_const_view& sp_view,
    const device::doublet_counter_container_types::const_view&
        doublet_counter_view,
    const doublet_container_view& mid_bot_doublet_view,
    const doublet_container_view& mid_top_doublet_view,
    const device::triplet_counter_container_types::const_view& tc_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        triplet_ps_view,
    triplet_container_view triplet_view) {

    // Check if anything needs to be done.
    const vecmem::device_vector<const prefix_sum_element_t> triplet_prefix_sum(
        triplet_ps_view);
    if (globalIndex >= triplet_prefix_sum.size()) {
        return;
    }

    // Get device copy of input parameters
    const device::doublet_counter_container_types::const_device
        doublet_counter_device(doublet_counter_view);
    const device_doublet_container mid_bot_doublet_device(mid_bot_doublet_view);
    const device_doublet_container mid_top_doublet_device(mid_top_doublet_view);
    const const_sp_grid_device sp_grid(sp_view);

    // Get the current work item
    const prefix_sum_element_t ps_idx = triplet_prefix_sum[globalIndex];
    const device::triplet_counter_container_types::const_device triplet_counts(
        tc_view);
    const device::triplet_counter mid_bot_counter =
        triplet_counts.get_items().at(ps_idx.first).at(ps_idx.second);
    const doublet mid_bot_doublet = mid_bot_counter.m_midBotDoublet;

    // middle spacepoint indexes
    const unsigned int spM_bin = mid_bot_doublet.sp1.bin_idx;
    const unsigned int spM_idx = mid_bot_doublet.sp1.sp_idx;
    // middle spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spM =
        sp_grid.bin(spM_bin)[spM_idx];

    // bottom spacepoint indexes
    const unsigned int spB_bin = mid_bot_doublet.sp2.bin_idx;
    const unsigned int spB_idx = mid_bot_doublet.sp2.sp_idx;
    // bottom spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spB =
        sp_grid.bin(spB_bin)[spB_idx];

    // Header of internal spacepoint container : spacepoint bin information
    // Item of internal spacepoint container : internal spacepoint objects
    // per bin
    const unsigned int num_compat_spM_per_bin =
        doublet_counter_device.get_headers().at(spM_bin).m_nSpM;

    // Header of doublet counter : number of compatible middle sp per bin
    // Item of doublet counter : doublet counter objects per bin
    // const
    // doublet_counter_container_types::const_device::item_vector::value_type
    // doublet_counter_per_bin =
    //     doublet_counter_device.get_items().at(spM_bin);
    const vecmem::device_vector<const doublet_counter> doublet_counter_per_bin =
        doublet_counter_device.get_items().at(spM_bin);

    // Header of doublet: number of mid_bot doublets per bin
    // Item of doublet: doublet objects per bin
    const unsigned int num_mid_bot_doublets_per_bin =
        mid_bot_doublet_device.get_headers().at(spM_bin).n_doublets;
    const vecmem::device_vector<const doublet> mid_bot_doublets_per_bin =
        mid_bot_doublet_device.get_items().at(spM_bin);

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin
    const vecmem::device_vector<const doublet> mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(spM_bin);

    // Set up the device result container
    device_triplet_container triplets(triplet_view);

    // Apply the conformal transformation to middle-bot doublet
    const traccc::lin_circle lb =
        doublet_finding_helper::transform_coordinates(spM, spB, true);

    // Calculate some physical quantities required for triplet compatibility
    // check
    const scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
    const scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2 *
                                       config.sigmaScattering *
                                       config.sigmaScattering;

    // These two quantities are used as output parameters in
    // triplet_finding_helper::isCompatible but their values are irrelevant
    scalar curvature, impact_parameter;

    // find the reference (start) index of the mid-top doublet container
    // item vector, where the doublets are recorded The start index is
    // calculated by accumulating the number of mid-top doublets of all
    // previous compatible middle spacepoints
    unsigned int mb_end_idx = 0;
    unsigned int mt_start_idx = 0;
    unsigned int mt_end_idx = 0;
    unsigned int mb_idx;

    // First, find the index of middle-bottom doublet
    for (unsigned int i = 0; i < num_mid_bot_doublets_per_bin; i++) {
        if (mid_bot_doublet == mid_bot_doublets_per_bin[i]) {
            mb_idx = i;
            break;
        }
    }

    for (unsigned int i = 0; i < num_compat_spM_per_bin; ++i) {
        mb_end_idx += doublet_counter_per_bin[i].m_nMidBot;
        mt_end_idx += doublet_counter_per_bin[i].m_nMidTop;

        if (mb_end_idx > mb_idx) {
            break;
        }
        mt_start_idx += doublet_counter_per_bin[i].m_nMidTop;
    }

    if (mt_end_idx >= mid_top_doublets_per_bin.size()) {
        mt_end_idx = std::min(mid_top_doublets_per_bin.size(), mt_end_idx);
    }

    if (mt_start_idx >= mid_top_doublets_per_bin.size()) {
        return;
    }

    // iterate over mid-top doublets
    for (unsigned int i = mt_start_idx; i < mt_end_idx; ++i) {
        const traccc::doublet mid_top_doublet = mid_top_doublets_per_bin[i];

        const unsigned int spT_bin = mid_top_doublet.sp2.bin_idx;
        const unsigned int spT_idx = mid_top_doublet.sp2.sp_idx;
        const traccc::internal_spacepoint<traccc::spacepoint> spT =
            sp_grid.bin(spT_bin)[spT_idx];

        // Apply the conformal transformation to middle-top doublet
        const traccc::lin_circle lt =
            doublet_finding_helper::transform_coordinates(spM, spT, false);

        // Check if mid-bot and mid-top doublets can form a triplet
        if (triplet_finding_helper::isCompatible(
                spM, lb, lt, config, iSinTheta2, scatteringInRegion2, curvature,
                impact_parameter)) {
            // Atomic reference for the triplet summary value for the bin of the
            // mid bottom doublet.
            vecmem::device_atomic_ref<unsigned int> num_triplets_per_bin(
                triplets.get_headers().at(spM_bin).n_triplets);
            num_triplets_per_bin.fetch_add(1);

            // Add triplet to jagged vector
            triplets.get_items().at(spM_bin).push_back(
                triplet({mid_bot_doublet.sp2, mid_bot_doublet.sp1,
                         mid_top_doublet.sp2, curvature,
                         -impact_parameter * filter_config.impactWeightFactor,
                         lb.Zo()}));
        }
    }
}

}  // namespace traccc::device
