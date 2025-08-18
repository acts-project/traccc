/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {
/**
 * @brief Kernel to create middle-bottom linearised circles.
 */
TRACCC_HOST_DEVICE
inline void make_mid_top_lincircles(
    global_index_t tid,
    device::device_doublet_collection_types::const_view mt_doublet_view,
    device::doublet_counter_collection_types::const_view doublet_count_view,
    edm::spacepoint_collection::const_view spacepoint_view,
    traccc::details::spacepoint_grid_types::const_view sp_grid_view,
    vecmem::data::vector_view<lin_circle> out_view) {

    const device::device_doublet_collection_types::const_device doublets(
        mt_doublet_view);
    const device::doublet_counter_collection_types::const_device doublet_counts(
        doublet_count_view);
    const edm::spacepoint_collection::const_device spacepoints(spacepoint_view);
    traccc::details::spacepoint_grid_types::const_device sp_grid(sp_grid_view);
    vecmem::device_vector<lin_circle> out(out_view);

    if (tid >= doublets.size()) {
        return;
    }

    const device::device_doublet dub = doublets.at(tid);
    const unsigned int counter_link = dub.counter_link;
    const device::doublet_counter count = doublet_counts.at(counter_link);
    const sp_location spM_loc = count.m_spM;
    const edm::spacepoint_collection::const_device::const_proxy_type spM =
        spacepoints.at(sp_grid.bin(spM_loc.bin_idx)[spM_loc.sp_idx]);
    const sp_location spT_loc = dub.sp2;
    const edm::spacepoint_collection::const_device::const_proxy_type spT =
        spacepoints.at(sp_grid.bin(spT_loc.bin_idx)[spT_loc.sp_idx]);

    out.at(tid) = doublet_finding_helper::transform_coordinates<
        traccc::details::spacepoint_type::top>(spM, spT);
}
}  // namespace traccc::device
