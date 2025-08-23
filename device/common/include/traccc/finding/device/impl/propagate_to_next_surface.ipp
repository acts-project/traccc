/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/propagator/constrained_step.hpp>
#include <detray/utils/tuple_helpers.hpp>

namespace traccc::device {

template <typename propagator_t, typename bfield_t>
TRACCC_HOST_DEVICE inline void propagate_to_next_surface(
    const global_index_t globalIndex, const finding_config& cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t>& payload) {

    using scalar_t = propagator_t::detector_type::scalar_type;

    if (globalIndex >= payload.n_in_params) {
        return;
    }

    // Theta id
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);

    const unsigned int param_id = param_ids.at(globalIndex);

    // Links
    vecmem::device_vector<const candidate_link> links(payload.links_view);

    const unsigned int link_idx = payload.prev_links_idx + param_id;
    const auto& link = links.at(link_idx);
    assert(link.step == payload.step);
    const unsigned int n_cands = link.step + 1 - link.n_skipped;

    // Parameter liveness
    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);

    // tips
    vecmem::device_vector<unsigned int> tips(payload.tips_view);
    vecmem::device_vector<unsigned int> tip_lengths(payload.tip_lengths_view);

    // Detector
    typename propagator_t::detector_type det(payload.det_data);

    // Parameters
    bound_track_parameters_collection_types::device params(payload.params_view);

    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    // Input bound track parameter
    const bound_track_parameters<> in_par = params.at(param_id);

    // Create propagator
    propagator_t propagator(cfg.propagation);

    // Create propagator state
    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    // Actor state
    // @TODO: simplify the syntax here
    // @NOTE: Post material interaction might be required here
    using actor_tuple_type =
        typename propagator_t::actor_chain_type::actor_tuple;
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
        s0{};
    typename detray::detail::tuple_element<3, actor_tuple_type>::type::state
        s3{};
    typename detray::detail::tuple_element<2, actor_tuple_type>::type::state s2{
        s3};
    typename detray::detail::tuple_element<4, actor_tuple_type>::type::state s4;

    typename detray::detail::tuple_element<5, actor_tuple_type>::type::state s5;
    s5.min_step_length = cfg.min_step_length_for_next_surface;
    s5.max_count = cfg.max_step_counts_for_next_surface;
    s4.min_pT(static_cast<scalar_t>(cfg.min_pT));
    s4.min_p(static_cast<scalar_t>(cfg.min_p));

    // Propagate to the next surface
    propagator.propagate(propagation, detray::tie(s0, s2, s3, s4, s5));

    // If a surface found, add the parameter for the next step
    if (s5.success) {
        assert(propagation._navigation.is_on_sensitive());
        assert(!propagation._stepping.bound_params().is_invalid());

        params[param_id] = propagation._stepping.bound_params();
        params_liveness[param_id] = 1u;
    } else {
        params_liveness[param_id] = 0u;

        if (n_cands >= cfg.min_track_candidates_per_track) {
            auto tip_pos = tips.push_back(link_idx);
            tip_lengths.at(tip_pos) = n_cands;
        }
    }
}

}  // namespace traccc::device
