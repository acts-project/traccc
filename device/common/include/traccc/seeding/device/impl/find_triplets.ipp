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
    const doublet_spM_container_types::const_view& mid_top_doublet_view,
    const device::triplet_counter_spM_container_types::const_view& tc_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& tc_ps_view,
    triplet_spM_container_types::view triplet_view) {

    // Check if anything needs to be done.
    const vecmem::device_vector<const prefix_sum_element_t> tc_prefix_sum(
        tc_ps_view);
    if (globalIndex >= tc_prefix_sum.size()) {
        return;
    }

    // Get device copy of input parameters
    const doublet_spM_container_types::const_device mid_top_doublet_device(
        mid_top_doublet_view);
    const const_sp_grid_device sp_grid(sp_view);

    // Get the current work item
    const prefix_sum_element_t ps_idx = tc_prefix_sum[globalIndex];
    const device::triplet_counter_spM_container_types::const_device
        triplet_counts(tc_view);

    const sp_location spM_loc =
        triplet_counts.get_headers().at(ps_idx.first).m_spM;
    const unsigned int num_triplets = triplet_counts.get_headers().at(ps_idx.first).m_nTriplets;
    const device::triplet_counter_spM mid_bot_counter =
        triplet_counts.get_items().at(ps_idx.first).at(ps_idx.second);
    const sp_location spB_loc = mid_bot_counter.m_spB;

    // middle spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spM =
        sp_grid.bin(spM_loc.bin_idx)[spM_loc.sp_idx];
    // bottom spacepoint
    const traccc::internal_spacepoint<traccc::spacepoint> spB =
        sp_grid.bin(spB_loc.bin_idx)[spB_loc.sp_idx];

    // Get mid top doublets for this bin
    const doublet_spM_collection_types::const_device mid_top_doublets_per_bin =
        mid_top_doublet_device.get_items().at(ps_idx.first);

    // Set up the device result container
    triplet_spM_container_types::device triplets(triplet_view);

    // We only need to update the spM information on the header once.
    if (ps_idx.second == 0) {
        triplets.get_headers().at(ps_idx.first) = {spM_loc, num_triplets};
    }

    // Apply the conformal transformation to middle-bot doublet
    const traccc::lin_circle lb = doublet_finding_helper::transform_coordinates<
        details::spacepoint_type::bottom>(spM, spB);

    // Calculate some physical quantities required for triplet compatibility
    // check
    const scalar iSinTheta2 = 1 + lb.cotTheta() * lb.cotTheta();
    const scalar scatteringInRegion2 = config.maxScatteringAngle2 * iSinTheta2 *
                                       config.sigmaScattering *
                                       config.sigmaScattering;

    // These two quantities are used as output parameters in
    // triplet_finding_helper::isCompatible but their values are irrelevant
    scalar curvature, impact_parameter;

    const unsigned int triplets_mb_begin = mid_bot_counter.posTriplets;
    const unsigned int triplets_mb_end =
        triplets_mb_begin + mid_bot_counter.m_nTriplets;
    // Find the reference (start) index where to fill these triplets into the
    // result.
    unsigned int posTriplets = triplets_mb_begin;

    const unsigned int num_top = mid_top_doublets_per_bin.size();

    // iterate over mid-top doublets
    for (unsigned int i = 0; i < num_top; ++i) {
        const sp_location spT_loc = mid_top_doublets_per_bin[i].sp2;

        const traccc::internal_spacepoint<traccc::spacepoint> spT =
            sp_grid.bin(spT_loc.bin_idx)[spT_loc.sp_idx];

        // Apply the conformal transformation to middle-top doublet
        const traccc::lin_circle lt =
            doublet_finding_helper::transform_coordinates<
                details::spacepoint_type::top>(spM, spT);

        // Check if mid-bot and mid-top doublets can form a triplet
        if (triplet_finding_helper::isCompatible(
                spM, lb, lt, config, iSinTheta2, scatteringInRegion2, curvature,
                impact_parameter)) {

            // Add triplet to jagged vector
            triplets.get_items().at(ps_idx.first).at(posTriplets++) =
                triplet_spM(
                    {spB_loc, spT_loc, curvature,
                     -impact_parameter * filter_config.impactWeightFactor,
                     lb.Zo(), triplets_mb_begin, triplets_mb_end});
        }
    }
}

}  // namespace traccc::device
