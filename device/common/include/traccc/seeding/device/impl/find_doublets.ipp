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
inline void find_doublets(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    const device::doublet_counter_container_types::const_view& doublet_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>&
        doublet_ps_view,
    doublet_container_types::view mb_doublets_view,
    doublet_container_types::view mt_doublets_view) {

    // Check if anything needs to be done.
    vecmem::device_vector<const prefix_sum_element_t> doublet_prefix_sum(
        doublet_ps_view);
    if (globalIndex >= doublet_prefix_sum.size()) {
        return;
    }

    // Get the middle spacepoint that we need to be looking at.
    const prefix_sum_element_t middle_sp_idx = doublet_prefix_sum[globalIndex];
    const device::doublet_counter_container_types::const_device doublet_counts(
        doublet_view);
    const device::doublet_counter_collection_types::const_device
        doublet_counts_in_bin =
            doublet_counts.get_items().at(middle_sp_idx.first);
    const device::doublet_counter middle_sp_counter =
        doublet_counts_in_bin.at(middle_sp_idx.second);

    // Set up the device containers.
    const const_sp_grid_device sp_grid(sp_view);
    doublet_container_types::device mb_doublets(mb_doublets_view);
    doublet_container_types::device mt_doublets(mt_doublets_view);

    // Get the doublet vectors just for the geometric bin of the middle
    // spacepoint.
    doublet_collection_types::device mb_doublets_in_bin =
        mb_doublets.get_items().at(middle_sp_counter.m_spM.bin_idx);
    doublet_collection_types::device mt_doublets_in_bin =
        mt_doublets.get_items().at(middle_sp_counter.m_spM.bin_idx);

    // Atomic references for the doublet summary values for the bin of the
    // middle spacepoint.
    vecmem::device_atomic_ref<unsigned int> mb_doublet_count(
        mb_doublets.get_headers()
            .at(middle_sp_counter.m_spM.bin_idx)
            .n_doublets);
    vecmem::device_atomic_ref<unsigned int> mt_doublet_count(
        mt_doublets.get_headers()
            .at(middle_sp_counter.m_spM.bin_idx)
            .n_doublets);

    // Get the spacepoint that we're evaluating in this thread, and treat that
    // as the "middle" spacepoint.
    const internal_spacepoint<spacepoint>& middle_sp =
        sp_grid.bin(middle_sp_counter.m_spM.bin_idx)
            .at(middle_sp_counter.m_spM.sp_idx);

    // Find the reference (start) index of the doublet container item vector,
    // where the doublets are recorded.
    const unsigned int mid_bot_start_idx = middle_sp_counter.m_posMidBot;
    const unsigned int mid_top_start_idx = middle_sp_counter.m_posMidTop;
    const unsigned int mid_top_end_idx =
        middle_sp_counter.m_posMidTop + middle_sp_counter.m_nMidTop;

    // The running indices for the middle-bottom and middle-top pairs.
    unsigned int mid_bot_idx = 0, mid_top_idx = 0;

    // The the IDs of the neighbouring bins along the phi and Z axes of the
    // grid.
    const detray::dindex_range phi_bins =
        sp_grid.axis_p0().range(middle_sp.phi(), config.neighbor_scope);
    const detray::dindex_range z_bins =
        sp_grid.axis_p1().range(middle_sp.z(), config.neighbor_scope);
    assert(z_bins[0] <= z_bins[1]);

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

            // Construct the "single index" that refers to this phi-Z bin.
            const unsigned int other_bin_idx =
                phi_bin + z_bin * sp_grid.axis_p0().bins();

            const unsigned int size = spacepoints.size();
            // Loop over all of those spacepoints.
            for (unsigned int other_sp_idx = 0; other_sp_idx < size;
                 ++other_sp_idx) {

                // Access the "other spacepoint".
                const internal_spacepoint<spacepoint> other_sp =
                    spacepoints.at(other_sp_idx);

                // Check if this spacepoint is a compatible "bottom" spacepoint
                // to the thread's "middle" spacepoint.
                if (doublet_finding_helper::isCompatible(middle_sp, other_sp,
                                                         config, true)) {

                    // Add it as a candidate to the middle-bottom container.
                    const unsigned int pos = mid_bot_start_idx + mid_bot_idx++;
                    assert(pos < mb_doublets_in_bin.size());
                    mb_doublets_in_bin.at(pos) = {
                        {static_cast<unsigned int>(
                             middle_sp_counter.m_spM.bin_idx),
                         static_cast<unsigned int>(
                             middle_sp_counter.m_spM.sp_idx)},
                        {other_bin_idx, other_sp_idx},
                        mid_top_start_idx,
                        mid_top_end_idx};
                    mb_doublet_count.fetch_add(1);
                }
                // Check if this spacepoint is a compatible "top" spacepoint to
                // the thread's "middle" spacepoint.
                if (doublet_finding_helper::isCompatible(middle_sp, other_sp,
                                                         config, false)) {

                    // Add it as a candidate to the middle-top container.
                    const unsigned int pos = mid_top_start_idx + mid_top_idx++;
                    assert(pos < mt_doublets_in_bin.size());
                    mt_doublets_in_bin.at(pos) = {
                        {static_cast<unsigned int>(
                             middle_sp_counter.m_spM.bin_idx),
                         static_cast<unsigned int>(
                             middle_sp_counter.m_spM.sp_idx)},
                        {other_bin_idx, other_sp_idx}};
                    mt_doublet_count.fetch_add(1);
                }
            }
        }
    }
}

}  // namespace traccc::device
