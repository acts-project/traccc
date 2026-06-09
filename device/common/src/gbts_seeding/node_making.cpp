/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/gbts_seeding/device/gbts_seeding_algorithm.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/vector.hpp>

// System include(s).
#include <algorithm>
#include <cmath>
#include <utility>

namespace traccc::device {

// Stage 1: node making
auto gbts_seeding_algorithm::make_nodes(
    const edm::spacepoint_collection::const_view& spacepoints,
    const edm::measurement_collection::const_view& measurements) const
    -> node_making_output {

    const gbts_seedfinder_config& cfg = m_config;
    const unsigned int nSp = copy().get_size(spacepoints);

    // 0. Bin spacepoints by the mapping supplied to config.surfaceToLayerMap.
    collection_types<unsigned int>::buffer layerCounts_buf(cfg.nLayers + 1,
                                                           mr().main);
    copy().memset(layerCounts_buf, 0)->ignore();

    collection_types<float4>::buffer reducedSP_buf(nSp, mr().main);
    copy().setup(reducedSP_buf)->ignore();

    collection_types<unsigned short>::buffer spacepointsLayer_buf(nSp,
                                                                  mr().main);
    copy().setup(spacepointsLayer_buf)->ignore();

    collection_types<short>::buffer volumeToLayerMap_buf(
        static_cast<unsigned int>(cfg.volumeToLayerMap.size()), mr().main);
    copy().setup(volumeToLayerMap_buf)->ignore();
    copy()(vecmem::get_data(cfg.volumeToLayerMap), volumeToLayerMap_buf)
        ->ignore();

    collection_types<std::pair<unsigned int, unsigned int>>::buffer
        surfaceToLayerMap_buf;
    if (!cfg.surfaceToLayerMap.empty()) {
        surfaceToLayerMap_buf =
            collection_types<std::pair<unsigned int, unsigned int>>::buffer(
                static_cast<unsigned int>(cfg.surfaceToLayerMap.size()),
                mr().main);
        copy().setup(surfaceToLayerMap_buf)->ignore();
        copy()(vecmem::get_data(cfg.surfaceToLayerMap), surfaceToLayerMap_buf)
            ->ignore();
    }

    collection_types<char>::buffer layerType_buf(cfg.nLayers, mr().main);
    copy().setup(layerType_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.type), layerType_buf)->ignore();
    count_sp_by_layer_kernel(
        {nSp, spacepoints, measurements, volumeToLayerMap_buf,
         surfaceToLayerMap_buf, layerType_buf, reducedSP_buf, layerCounts_buf,
         spacepointsLayer_buf, cfg.volumeToLayerMap.size(),
         cfg.surfaceToLayerMap.size(), cfg.sp_counting_params});

    // CPU: prefix-sum the per-layer counts on the host (turns counts into the
    // layer-ordered write offsets used by bin_sp).
    collection_types<unsigned int>::host layerCounts(cfg.nLayers + 1,
                                                     mr().host);
    copy()(vecmem::get_data(layerCounts_buf), layerCounts)->wait();
    for (unsigned int layer = 0; layer < cfg.nLayers; layer++) {
        layerCounts[layer + 1] += layerCounts[layer];
    }
    copy()(vecmem::get_data(layerCounts), layerCounts_buf)->wait();

    const unsigned int nNodes =
        static_cast<unsigned int>(layerCounts[cfg.nLayers]);
    TRACCC_DEBUG("nNodes " << nNodes);
    if (nNodes == 0) {
        TRACCC_WARNING("No nodes were found after spacepoint counting");
        return node_making_output{};
    }

    collection_types<float4>::buffer sp_params_buf(nSp, mr().main);
    copy().setup(sp_params_buf)->ignore();
    collection_types<unsigned int>::buffer original_sp_idx_buf(nSp, mr().main);
    copy().setup(original_sp_idx_buf)->ignore();

    // 1. Fused binning: scatter spacepoints into layer-ordered slots, compute
    //    their eta/phi bin indices and fill the (eta, phi) histogram, all in a
    //    single pass.
    collection_types<std::pair<unsigned int, unsigned int>>::buffer
        layer_info_buf(cfg.nLayers, mr().main);
    copy().setup(layer_info_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.info), layer_info_buf)->ignore();

    collection_types<std::pair<float, float>>::buffer layer_geo_buf(cfg.nLayers,
                                                                    mr().main);
    copy().setup(layer_geo_buf)->ignore();
    copy()(vecmem::get_data(cfg.layerInfo.geo), layer_geo_buf)->ignore();

    collection_types<unsigned int>::buffer node_phi_index_buf(nNodes,
                                                              mr().main);
    copy().setup(node_phi_index_buf)->ignore();

    collection_types<unsigned int>::buffer node_eta_index_buf(nNodes,
                                                              mr().main);
    copy().setup(node_eta_index_buf)->ignore();

    const unsigned int hist_size = cfg.n_eta_bins * cfg.n_phi_bins;
    collection_types<unsigned int>::buffer eta_phi_histo_buf(hist_size,
                                                             mr().main);
    copy().setup(eta_phi_histo_buf)->ignore();
    copy().memset(eta_phi_histo_buf, 0)->ignore();
    collection_types<unsigned int>::buffer phi_cusums_buf(hist_size, mr().main);
    copy().setup(phi_cusums_buf)->ignore();

    bin_sp_kernel({nSp, cfg.n_phi_bins, sp_params_buf, reducedSP_buf,
                   layerCounts_buf, spacepointsLayer_buf, original_sp_idx_buf,
                   layer_info_buf, layer_geo_buf, node_eta_index_buf,
                   node_phi_index_buf, eta_phi_histo_buf});

    collection_types<unsigned int>::buffer eta_node_counter_buf(cfg.n_eta_bins,
                                                                mr().main);
    copy().setup(eta_node_counter_buf)->ignore();

    eta_phi_counting_kernel({cfg.n_eta_bins, cfg.n_phi_bins, eta_phi_histo_buf,
                             eta_node_counter_buf, phi_cusums_buf});

    // CPU: prefix-sum the per-eta counts and build the (begin, end) node range
    // per eta bin on the host.
    collection_types<unsigned int>::host eta_sums(cfg.n_eta_bins, mr().host);
    copy()(vecmem::get_data(eta_node_counter_buf), eta_sums)->wait();
    for (unsigned int k = 0; k < cfg.n_eta_bins; k++) {
        eta_sums[k + 1] += eta_sums[k];
    }
    copy()(vecmem::get_data(eta_sums), eta_node_counter_buf)->wait();

    collection_types<unsigned int>::host eta_bin_views(2 * cfg.n_eta_bins,
                                                       mr().host);
    for (unsigned int view_idx = 0; view_idx < cfg.n_eta_bins; view_idx++) {
        const unsigned int pos = 2 * view_idx;
        eta_bin_views[pos] = (view_idx == 0) ? 0 : eta_sums[view_idx - 1];
        eta_bin_views[pos + 1] = eta_sums[view_idx];
    }

    eta_phi_prefix_sum_kernel(
        {cfg.n_eta_bins, cfg.n_phi_bins, eta_node_counter_buf, phi_cusums_buf});

    collection_types<float4>::buffer node_params_buf(nNodes, mr().main);
    copy().setup(node_params_buf)->ignore();
    collection_types<float>::buffer node_phi_buf(nNodes, mr().main);
    copy().setup(node_phi_buf)->ignore();
    collection_types<unsigned int>::buffer node_index_buf(nNodes, mr().main);
    copy().setup(node_index_buf)->ignore();

    // Optional tau LUT consumed by device::node_sorting when
    // cfg.node_sorting.useTauLUT is set. A size-1 dummy is allocated when the
    // LUT is unused so the kernel always receives a valid (never-read) view.
    const unsigned int tau_lut_size = std::max<unsigned int>(
        1u, static_cast<unsigned int>(cfg.tau_lut.size()));
    collection_types<float>::buffer tau_lut_buf(tau_lut_size, mr().main);
    copy().setup(tau_lut_buf)->ignore();
    if (!cfg.tau_lut.empty()) {
        copy()(vecmem::get_data(cfg.tau_lut), tau_lut_buf)->ignore();
    }

    node_sorting_kernel({nNodes, cfg.n_phi_bins, sp_params_buf,
                         node_eta_index_buf, node_phi_index_buf, phi_cusums_buf,
                         node_params_buf, node_phi_buf, node_index_buf,
                         original_sp_idx_buf, tau_lut_buf, cfg.node_sorting});

    collection_types<unsigned int>::buffer eta_bin_views_buf(2 * cfg.n_eta_bins,
                                                             mr().main);
    copy().setup(eta_bin_views_buf)->ignore();
    copy()(vecmem::get_data(eta_bin_views), eta_bin_views_buf)->wait();

    collection_types<float>::buffer bin_rads_buf(2 * cfg.n_eta_bins, mr().main);
    copy().setup(bin_rads_buf)->ignore();

    minmax_rad_kernel(
        {cfg.n_eta_bins, eta_bin_views_buf, node_params_buf, bin_rads_buf});

    collection_types<float>::host bin_rads(2 * cfg.n_eta_bins, mr().host);
    copy()(vecmem::get_data(bin_rads_buf), bin_rads)->wait();

    return node_making_output{std::move(reducedSP_buf),
                              std::move(node_params_buf),
                              std::move(node_phi_buf),
                              std::move(node_index_buf),
                              std::move(bin_rads),
                              std::move(eta_bin_views),
                              nNodes};
}
}  // namespace traccc::device
