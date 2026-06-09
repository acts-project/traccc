/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <cstdint>
#include <utility>

namespace traccc::device {

/// @brief Fit each candidate path and emit seed proposals that pass quality
/// cuts.
///
/// One thread per path-store entry walks backwards from a leaf, gathers the
/// involved spacepoints, runs the helix / chi-squared fit, and on success
/// atomically claims a slot in d_seed_proposals_view, bids for its leaf
/// edge via d_edge_bids_view, and tags ambiguity.
///
/// @param[in]  globalIndex                 Current thread index
/// @param[in]  d_sp_reduced_view           Reduced (x, y, z, r) per SP
/// @param[in]  d_output_graph_view         Compact graph
/// @param[in]  d_path_store_view           Per-path (parent, edge) entries
/// @param[out] d_seed_proposals_view       (path_store index, level) per seed
/// @param[in,out] d_edge_bids_view         Per-edge highest-bidder seed
/// @param[out] d_seed_ambiguity_view       Per-seed ambiguity tag
/// @param[in,out] nPropsCounter            Global atomic count of proposals
/// @param[in]  nTerminusEdges              Number of terminus edges
/// @param[in]  minLevel                    Minimum required path length
/// @param[in]  max_num_neighbours          Maximum neighbours per edge
/// @param[in]  seed_extraction_params      Fit / chi-squared / curvature cuts
/// @param[in]  max_z0                      Max |z0| at the beamline for
/// extrapolation cuts
///
TRACCC_HOST_DEVICE
inline void fit_segments(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_reduced_view,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    unsigned int& nPropsCounter, const unsigned int nTerminusEdges,
    const unsigned char minLevel, const unsigned int max_num_neighbours,
    const gbts_seed_extraction_params& seed_extraction_params,
    const float max_z0);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/fit_segments.ipp"
