/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/finding/actors/measurement_kalman_updater.hpp"
#include "traccc/finding/details/combinatorial_kalman_filter_types.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/measurement_selector.hpp"
#include "traccc/sanity/contiguous_on.hpp"
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"
#include "traccc/utils/prob.hpp"
#include "traccc/utils/propagation.hpp"

// Detray include(s).
#include <detray/plugins/algebra/array_definitions.hpp>
#include <detray/utils/tuple_helpers.hpp>

namespace traccc::device {

template <typename propagator_t>
TRACCC_HOST_DEVICE inline void kalman_track_follower(
    const global_index_t globalIndex, const finding_config& cfg,
    const kalman_track_follower_payload<propagator_t>& payload) {

    using detector_t = typename propagator_t::detector_type;
    using algebra_t = typename detector_t::algebra_type;
    using scalar_t = detray::dscalar<algebra_t>;

    if (globalIndex >= payload.seeds_view.size()) {
        return;
    }

    // Detector
    detector_t det(payload.det_data);

    // Parameters
    bound_track_parameters_collection_types::device seeds(payload.seeds_view);

    // Input bound track parameter
    const bound_track_parameters<algebra_t> seed = seeds.at(globalIndex);

    // Create propagator
    auto prop_cfg{cfg.propagation};
    propagator_t propagator(prop_cfg);

    // Create propagator state
    typename propagator_t::state propagation(seed, payload.field_data, det);
    propagation.set_particle(
        detail::correct_particle_hypothesis(cfg.ptc_hypothesis, seed));

    // Actor state
    // @TODO: simplify the syntax here
    // @NOTE: Post material interaction might be required here
    /*using actor_tuple_type =
        typename propagator_t::actor_chain_type::actor_tuple;
    // Pathlimit aborter
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
        s0{};
    typename detray::detail::tuple_element<1, actor_tuple_type>::type::state
        s1{};
    // CKF-interactor
    typename detray::detail::tuple_element<3, actor_tuple_type>::type::state
        s3{};
    // Interaction register
    typename detray::detail::tuple_element<2, actor_tuple_type>::type::state s2{
        s3};
    // Parameter resetter
    typename detray::detail::tuple_element<4, actor_tuple_type>::type::state s4{
        prop_cfg};
    // Momentum aborter
    typename detray::detail::tuple_element<5, actor_tuple_type>::type::state s5;
    // CKF aborter
    typename detray::detail::tuple_element<6, actor_tuple_type>::type::state s6;

    s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
    s5.min_p(static_cast<scalar_t>(cfg.min_p));
    s6.min_step_length = cfg.min_step_length_for_next_surface;
    s6.max_count = cfg.max_step_counts_for_next_surface;

    // Propagate to the next surface
    propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));

    // If a surface found, add the parameter for the next step
    if (s6.success) {
        assert(propagation._navigation.is_on_sensitive());
        assert(!propagation._stepping.bound_params().is_invalid());

        params[param_id] = propagation._stepping.bound_params();
        params_liveness[param_id] = 1u;

        const scalar theta = params[param_id].theta();
        if (theta <= 0.f || theta >= 2.f * constant<traccc::scalar>::pi) {
            TRACCC_ERROR_DEVICE("Theta is zero after propagation");
            params_liveness[param_id] = 0u;
        }

        if (!std::isfinite(params[param_id].phi())) {
            TRACCC_ERROR_DEVICE(
                "Phi is infinite after propagation (Matrix inversion)");
            params_liveness[param_id] = 0u;
        }

        if (math::fabs(params[param_id].qop()) == 0.f) {
            TRACCC_ERROR_DEVICE("q/p is zero after propagation");
            params_liveness[param_id] = 0u;
        }
    } else {
        params_liveness[param_id] = 0u;
    }

    if (params_liveness[param_id] == 0 &&
        n_cands >= cfg.min_track_candidates_per_track) {
        TRACCC_VERBOSE_DEVICE("Create tip: No next sensitive found");
        auto tip_pos = tips.push_back(link_idx);
        tip_lengths.at(tip_pos) = n_cands;
    }*/
}

}  // namespace traccc::device
