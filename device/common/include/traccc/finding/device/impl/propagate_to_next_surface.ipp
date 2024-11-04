/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/core/detail/tuple_container.hpp"
#include "detray/propagator/constrained_step.hpp"
#include "detray/utils/tuple.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_parameters.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/utils/particle.hpp"

namespace traccc::device {

template <typename propagator_t, typename bfield_t, typename config_t>
TRACCC_DEVICE inline void propagate_to_next_surface(
    std::size_t globalIndex, const config_t cfg,
    const propagate_to_next_surface_payload<propagator_t, bfield_t> &payload) {

    if (globalIndex >= payload.n_in_params) {
        return;
    }

    // Theta id
    vecmem::device_vector<const unsigned int> param_ids(payload.param_ids_view);

    const unsigned int param_id = param_ids.at(globalIndex);

    // Number of tracks per seed
    vecmem::device_vector<unsigned int> n_tracks_per_seed(
        payload.n_tracks_per_seed_view);

    // Links
    vecmem::device_vector<const candidate_link> links(payload.links_view);

    // Seed id
    unsigned int orig_param_id = links.at(param_id).seed_idx;

    // Count the number of tracks per seed
    vecmem::device_atomic_ref<unsigned int> num_tracks_per_seed(
        n_tracks_per_seed.at(orig_param_id));

    const unsigned int s_pos = num_tracks_per_seed.fetch_add(1);
    vecmem::device_vector<unsigned int> params_liveness(
        payload.params_liveness_view);

    if (s_pos >= cfg.max_num_branches_per_seed) {
        params_liveness[param_id] = 0u;
        return;
    }

    // tips
    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        payload.tips_view);

    if (links.at(param_id).n_skipped > cfg.max_num_skipping_per_cand) {
        params_liveness[param_id] = 0u;
        tips.push_back({payload.step, param_id});
        return;
    }

    // Detector
    typename propagator_t::detector_type det(payload.det_data);

    // Parameters
    bound_track_parameters_collection_types::device params(payload.params_view);

    if (params_liveness.at(param_id) == 0u) {
        return;
    }

    // Input bound track parameter
    const bound_track_parameters in_par = params.at(param_id);

    // Create propagator
    propagator_t propagator(cfg.propagation);

    // Create propagator state
    typename propagator_t::state propagation(in_par, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, in_par));
    propagation._stepping
        .template set_constraint<detray::step::constraint::e_accuracy>(
            cfg.propagation.stepping.step_constraint);

    // Generate actor states and references to the states
    using actor_chain_t = typename propagator_t::actor_chain_type;
    auto [actors_states, actor_chain_state] =
        actor_chain_t::make_actor_states();

    auto &path_abrt_state =
        detray::detail::get<detray::pathlimit_aborter::state>(actors_states);
    path_abrt_state.set_path_limit(cfg.propagation.stepping.path_limit);

    auto &ckf_abrt_state =
        detray::detail::get<ckf_aborter::state>(actors_states);
    ckf_abrt_state.min_step_length = cfg.min_step_length_for_next_surface;
    ckf_abrt_state.max_count = cfg.max_step_counts_for_next_surface;

    // @TODO: Should be removed once detray is fixed to set the volume in the
    // constructor
    propagation._navigation.set_volume(in_par.surface_link().volume());

    // Propagate to the next surface
    propagator.propagate_sync(propagation, actor_chain_state);

    // If a surface found, add the parameter for the next step
    if (ckf_abrt_state.success) {
        params[param_id] = propagation._stepping.bound_params();

        if (payload.step == cfg.max_track_candidates_per_track - 1) {
            tips.push_back({payload.step, param_id});
            params_liveness[param_id] = 0u;
        } else {
            params_liveness[param_id] = 1u;
        }
    } else {
        params_liveness[param_id] = 0u;

        if (payload.step >= cfg.min_track_candidates_per_track - 1) {
            tips.push_back({payload.step, param_id});
        }
    }
}

}  // namespace traccc::device
