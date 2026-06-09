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
#include "traccc/edm/seed_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

/// @brief Convert accepted seed proposals into 3-spacepoint edm::seeds.
///
/// Each thread (strided by gridSize) processes one accepted proposal,
/// reads the three constituent spacepoints from d_sp_params_view, applies
/// dropout / curvature / hit-fraction cuts, and appends a seed to the output
/// resizable buffer on success.
///
/// @param[in]  globalIndex                 Proposal index processed by this
/// call
/// @param[in]  d_seed_proposals_view       Per-seed (path index, level)
/// @param[in]  d_seed_ambiguity_view       Per-seed ambiguity tag
/// @param[in]  d_path_store_view           Per-path (parent, edge) entries
/// @param[in]  d_output_graph_view         Compact graph
/// @param[in]  d_sp_params_view            Layer-ordered (x, y, z, r) per SP
/// @param[out] output_seeds                Appended 3-SP edm::seed records
/// @param[in]  d_hit_bids_view             Per-hit highest-bidder seed
/// @param[in]  max_num_neighbours          Maximum neighbours per edge
/// @param[in]  dcurv_cut_m                 Curvature-difference dropout cut
/// @param[in]  force_dropout_max_curv_m    Hard curvature cutoff for dropout
/// @param[in]  best_hit_frac               Minimum best-bid fraction to keep
/// @param[in]  tight_bid_cot_threshold     Tight-vs-loose cot(theta) threshold
/// @param[in]  use_dropout                 Master switch for hit-bid dropout
///
TRACCC_HOST_DEVICE
inline void gbts_seed_conversion(
    const global_index_t globalIndex,
    const collection_types<int2>::const_view& d_seed_proposals_view,
    const collection_types<char>::const_view& d_seed_ambiguity_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<float4>::const_view& d_sp_params_view,
    const edm::seed_collection::view& output_seeds,
    const collection_types<unsigned long long int>::view& d_hit_bids_view,
    const unsigned int max_num_neighbours, const float dcurv_cut_m,
    const float force_dropout_max_curv_m, const float best_hit_frac,
    const float tight_bid_cot_threshold, const bool use_dropout);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_seed_conversion.ipp"
