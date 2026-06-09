/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <array>
#include <cmath>
#include <cstdint>

namespace traccc::device {

namespace gbts_detail {

struct Tracklet {
    unsigned int nodes[traccc::device::gbts_consts::max_cca_iter + 1];
    int size;
};

TRACCC_HOST_DEVICE inline traccc::float2 estimate_params(
    const std::array<traccc::float4, 3>& sps) {

    float u[2], v[2];

    const float x0 = sps[1].x;
    const float y0 = sps[1].y;
    const float r0 = sqrtf(x0 * x0 + y0 * y0);
    const float cosA = x0 / r0;
    const float sinA = y0 / r0;

    for (unsigned int k = 0; k < 2; k++) {
        const unsigned int sp_idx = (k == 1) ? 2u : k;
        const float dx = sps[sp_idx].x - x0;
        const float dy = sps[sp_idx].y - y0;
        const float r2_inv = 1.0f / (dx * dx + dy * dy);
        const float xn = dx * cosA + dy * sinA;
        const float yn = -dx * sinA + dy * cosA;
        u[k] = xn * r2_inv;
        v[k] = yn * r2_inv;
    }

    const float du = u[0] - u[1];
    if (du == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }
    const float A = (v[0] - v[1]) / du;
    const float B = v[1] - A * u[1];
    const float curv =
        1000.0f * B / sqrtf(1 + A * A);  // Curvature from mm^-1 to m^-1
    const float cot_t = (sps[2].z - sps[1].z) /
                        (sqrtf(sps[2].x * sps[2].x + sps[2].y * sps[2].y) - r0);
    return make_float2(curv, cot_t);
}

}  // namespace gbts_detail

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
    const float tight_bid_cot_threshold, const bool use_dropout) {

    edm::seed_collection::device seeds_device(output_seeds);
    const collection_types<int2>::const_device d_seed_proposals(
        d_seed_proposals_view);
    const collection_types<char>::const_device d_seed_ambiguity(
        d_seed_ambiguity_view);
    const collection_types<int2>::const_device d_path_store(d_path_store_view);
    const collection_types<unsigned int>::const_device d_output_graph(
        d_output_graph_view);
    const collection_types<float4>::const_device d_sp_params(d_sp_params_view);
    collection_types<unsigned long long int>::device d_hit_bids(
        d_hit_bids_view);

    // Row-major output graph: each edge owns a contiguous block of
    // edge_size = 2 + 1 + max_num_neighbours ints.
    const unsigned int edge_size = 2u + 1u + max_num_neighbours;

    // One proposal per call; the grid-stride loop lives in the kernel wrapper.
    const unsigned int prop_idx = globalIndex;
    if (d_seed_ambiguity[prop_idx] == -2) {
        return;
    }
    char best_for_hit = 0;
    gbts_detail::Tracklet seed;
    seed.size = 0;
    const int2 prop = d_seed_proposals[prop_idx];
    int2 path = make_int2(0, prop.y);
    while (path.y >= 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        seed.nodes[seed.size++] =
            d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                           gbts_consts::node1];
        best_for_hit += (prop_idx == (d_hit_bids[seed.nodes[seed.size - 1]] &
                                      0xFFFFFFFFLL));
    }
    seed.nodes[seed.size++] =
        d_output_graph[edge_size * static_cast<unsigned int>(path.x) +
                       gbts_consts::node2];
    best_for_hit +=
        (prop_idx == (d_hit_bids[seed.nodes[seed.size - 1]] & 0xFFFFFFFFLL));

    if (best_for_hit < best_hit_frac * static_cast<float>(seed.size)) {
        return;
    }
    char diff_code = 0;
    bool force_dropout = false;
    if (use_dropout) {
        std::array<traccc::float4, 3> sps = {
            d_sp_params[seed.nodes[seed.size - 1]],
            d_sp_params[seed.nodes[(seed.size - 1) / 2 + 1]],
            d_sp_params[seed.nodes[0]]};
        const traccc::float2 curv_cot_1 = gbts_detail::estimate_params(sps);
        sps[1] = d_sp_params[seed.nodes[(seed.size - 1) / 2]];
        const traccc::float2 curv_cot_2 = gbts_detail::estimate_params(sps);
        sps[0] = d_sp_params[seed.nodes[seed.size - 2]];
        const traccc::float2 curv_cot_3 = gbts_detail::estimate_params(sps);
        if ((best_for_hit < seed.size - 1) &
            (fabsf(curv_cot_1.y + curv_cot_2.y +
                   curv_cot_3.y) <  // Checking against the average cot(theta)
                                    // of the three tracklets
             3.0f * tight_bid_cot_threshold) &
            (seed.size < 5)) {  // Don't apply dropout to seeds of length 5 or
                                // more. To avoid dropping good seeds.
            return;
        }
        std::array<float, 3> diff = {fabsf(curv_cot_1.x - curv_cot_2.x),
                                     fabsf(curv_cot_2.x - curv_cot_3.x),
                                     fabsf(curv_cot_1.x - curv_cot_3.x)};
        diff_code = static_cast<char>(4 * (diff[0] < dcurv_cut_m) +
                                      2 * (diff[1] < dcurv_cut_m) +
                                      (diff[2] < dcurv_cut_m));
        force_dropout = fabsf(curv_cot_1.x + curv_cot_2.x + curv_cot_3.x) <
                        3.0f * force_dropout_max_curv_m;
        force_dropout |= (fabsf(curv_cot_1.y + curv_cot_2.y + curv_cot_3.y) <
                          3.0f * tight_bid_cot_threshold) &
                         (diff_code == 0);
    }
    float quality = static_cast<float>(prop.x);
    // use one seed from a consistant pair/set + the inconsistant one
    // sample spacepoints from tracklet to create seeds
    // include 1st order unless either 2 or 3 are consitant with the other
    // and 1
    if (diff_code != 3 & diff_code != 6 | force_dropout) {
        seeds_device.push_back({seed.nodes[seed.size - 1],
                                seed.nodes[(seed.size - 1) / 2 + 1],
                                seed.nodes[0], quality});
    }
    // include 2nd order if it consistant with 1 and 3 or only 1 and 3 are
    // consistant
    if (diff_code == 1 | diff_code == 6) {
        seeds_device.push_back({seed.nodes[seed.size - 1],
                                seed.nodes[(seed.size - 1) / 2], seed.nodes[0],
                                quality});
    }
    // include 3rd order if it is consistant with 1 and 2 or only 1 and 2
    // are consistant or if only 2 and 3 are consistant
    if (diff_code == 2 | diff_code == 3 | diff_code == 4 | force_dropout) {
        seeds_device.push_back({seed.nodes[seed.size - 2],
                                seed.nodes[(seed.size - 1) / 2], seed.nodes[0],
                                quality});
    }
}

}  // namespace traccc::device
