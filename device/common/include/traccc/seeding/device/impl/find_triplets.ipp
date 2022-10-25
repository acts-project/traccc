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
inline void find_triplets(
    const std::size_t globalIndex, const seedfinder_config& config,
    const seedfilter_config& filter_config, const sp_grid_const_view& sp_view,
    const doublet_container_types::const_view& mid_top_doublet_view,
    const device::triplet_counter_container_types::const_view& tc_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        triplet_ps_view,
    triplet_container_types::view triplet_view) {

    // Check if anything needs to be done.
    const vecmem::device_vector<const prefix_sum_element_t> triplet_prefix_sum(
        triplet_ps_view);
    if (globalIndex >= triplet_prefix_sum.size()) {
        return;
    }

    // Get device copy of input parameters
    const doublet_container_types::const_device mid_top_doublet_device(
        mid_top_doublet_view);
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

    // Header of doublet: number of mid_top doublets per bin
    // Item of doublet: doublet objects per bin
    const vecmem::device_vector<const doublet> mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(spM_bin);

    // Set up the device result container
    triplet_container_types::device triplets(triplet_view);

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
    // item vector, where the doublets are recorded
    const unsigned int mt_start_idx = mid_bot_counter.m_mt_start_idx;
    const unsigned int mt_end_idx = mid_bot_counter.m_mt_end_idx;
    const unsigned int triplets_mb_begin = mid_bot_counter.posTriplets;
    const unsigned int triplets_mb_end =
        triplets_mb_begin + mid_bot_counter.m_nTriplets;
    unsigned int posTriplets = triplets_mb_begin;

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
            triplets.get_items().at(spM_bin).at(posTriplets++) =
                triplet({mid_bot_doublet.sp2, mid_bot_doublet.sp1,
                         mid_top_doublet.sp2, curvature,
                         -impact_parameter * filter_config.impactWeightFactor,
                         lb.Zo(), triplets_mb_begin, triplets_mb_end});
        }
    }
}

}  // namespace traccc::device
