/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/doublet_finding_helper.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cassert>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_doublets(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& sp_ps_view,
    doublet_counter_container_types::view doublet_view) {

    // Check if anything needs to be done.
    vecmem::device_vector<const prefix_sum_element_t> sp_prefix_sum(sp_ps_view);
    if (globalIndex >= sp_prefix_sum.size()) {
        return;
    }

    // Get the middle spacepoint that we need to be looking at.
    const prefix_sum_element_t middle_sp_idx = sp_prefix_sum[globalIndex];

    // Set up the device containers.
    const const_sp_grid_device sp_grid(sp_view);
    doublet_counter_container_types::device doublet_counter(doublet_view);

    // Get the spacepoint that we're evaluating in this thread, and treat that
    // as the "middle" spacepoint.
    const internal_spacepoint<spacepoint>& middle_sp =
        sp_grid.bin(middle_sp_idx.first).at(middle_sp_idx.second);

    // The the IDs of the neighbouring bins along the phi and Z axes of the
    // grid.
    const detray::dindex_range phi_bins =
        sp_grid.axis_p0().range(middle_sp.phi(), config.neighbor_scope);
    const detray::dindex_range z_bins =
        sp_grid.axis_p1().range(middle_sp.z(), config.neighbor_scope);
    assert(z_bins[0] <= z_bins[1]);

    // The number of middle-bottom candidates found for this thread's middle
    // spacepoint.
    unsigned int n_mb_cand = 0;
    // The number of middle-top candidates found for this thread's middle
    // spacepoint.
    unsigned int n_mt_cand = 0;

    // Iterate over all of the neighboring phi bins, including the same bin that
    // the middle spacepoint is in. The loop over the phi bins needs to take
    // into account that we may iterate over the "wrap around point" of the
    // axis.
    for (detray::dindex phi_bin_iterator = phi_bins[0];
         phi_bin_iterator <=
         (phi_bins[1] +
          (phi_bins[0] > phi_bins[1] ? sp_grid.axis_p0().n_bins : 0));
         ++phi_bin_iterator) {

        // Set up the phi bin index that we are actually meant to use inside of
        // the loop. We could also use a modulo operation here, but that would
        // be slightly more expensive in this specific case.
        const detray::dindex phi_bin =
            (phi_bin_iterator >= sp_grid.axis_p0().n_bins
                 ? phi_bin_iterator - sp_grid.axis_p0().n_bins
                 : phi_bin_iterator);

        // Iterate over all of the neighboring Z bins, including the same bin
        // that the middle spacepoint is in. This is a much easier iteration, as
        // the Z axis does not "wrap around".
        for (detray::dindex z_bin = z_bins[0]; z_bin <= z_bins[1]; ++z_bin) {

            // Ask the grid for all of the spacepoints in this specific bin.
            typename const_sp_grid_device::serialized_storage::const_reference
                spacepoints = sp_grid.bin(phi_bin, z_bin);

            // Loop over all of those spacepoints.
            for (const internal_spacepoint<spacepoint>& other_sp :
                 spacepoints) {

                // Check if this spacepoint is a compatible "bottom" spacepoint
                // to the thread's "middle" spacepoint.
                if (doublet_finding_helper::isCompatible(middle_sp, other_sp,
                                                         config, true)) {
                    ++n_mb_cand;
                }
                // Check if this spacepoint is a compatible "top" spacepoint to
                // the thread's "middle" spacepoint.
                if (doublet_finding_helper::isCompatible(middle_sp, other_sp,
                                                         config, false)) {
                    ++n_mt_cand;
                }
            }
        }
    }

    // Add the counts if compatible bottom *AND* top candidates were found for
    // the middle spacepoint in question.
    if ((n_mb_cand > 0) && (n_mt_cand > 0)) {

        // Increment the summary values in the header object.
        doublet_counter_header& header =
            doublet_counter.get_headers().at(middle_sp_idx.first);
        vecmem::device_atomic_ref<unsigned int> nSpM(header.m_nSpM);
        nSpM.fetch_add(1);
        vecmem::device_atomic_ref<unsigned int> nMidBot(header.m_nMidBot);
        const unsigned int posBot = nMidBot.fetch_add(n_mb_cand);
        vecmem::device_atomic_ref<unsigned int> nMidTop(header.m_nMidTop);
        const unsigned int posTop = nMidTop.fetch_add(n_mt_cand);

        // Add the number of candidates for the "current bin".
        doublet_counter.get_items()
            .at(middle_sp_idx.first)
            .push_back({{static_cast<unsigned int>(middle_sp_idx.first),
                         static_cast<unsigned int>(middle_sp_idx.second)},
                        n_mb_cand,
                        n_mt_cand,
                        posBot,
                        posTop});
    }
}

}  // namespace traccc::device
