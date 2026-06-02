/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/logging.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/geometry/shapes/line.hpp>
#include <detray/geometry/tracking_surface.hpp>
#include <detray/materials/material_rod.hpp>
#include <detray/navigation/direct_navigator.hpp>
#include <detray/navigation/external_surface.hpp>
#include <detray/plugins/algebra/array_definitions.hpp>
#include <detray/propagator/actor_chain.hpp>
#include <detray/propagator/constrained_step.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/utils/ranges/single.hpp>
#include <detray/utils/tuple_helpers.hpp>

// System include(s).
#include <limits>

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

    // Expected layer patterns
    vecmem::device_vector<expected_layer_pattern_type>
        expected_layer_patterns(payload.expected_layer_patterns_view);

    // tips
    vecmem::device_vector<unsigned int> tips(payload.tips_view);
    vecmem::device_vector<unsigned int> tip_lengths(payload.tip_lengths_view);

    // Detector
    typename propagator_t::detector_type det(payload.det_data);

    // Parameters
    bound_track_parameters_collection_types::device params(payload.params_view);

    if (params_liveness.at(param_id) == 0u) {
        //NOTE: FIX FOR GARABEGE VALUES OF EXPECTED LAYER PATTERNS. Set to 0 if param is dead, otherwise 1.
        if (expected_layer_patterns.size() > link_idx) {
            auto& link_pattern = expected_layer_patterns.at(link_idx);
            for (unsigned int i = 0u; i < link_pattern.size(); ++i) {
                link_pattern[i] = 0u;
            }
        }
        return;
    }

    // Input bound track parameter
    const bound_track_parameters<> in_par = params.at(param_id);

    // Create propagator
    auto prop_cfg{cfg.propagation};
    prop_cfg.navigation.estimate_scattering_noise = false;
    propagator_t propagator(prop_cfg);

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
    // Pathlimit aborter
    typename detray::detail::tuple_element<0, actor_tuple_type>::type::state
        s0{};
    
    // typename detray::detail::tuple_element<1, actor_tuple_type>::type::state
    //     s1{};
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
    // Expected layer pattern collector
    typename detray::detail::tuple_element<6, actor_tuple_type>::type::state s6;
    // CKF aborter
    typename detray::detail::tuple_element<7, actor_tuple_type>::type::state s7;

    /*
     * If we are running the MBF smoother, we need to accumulate the Jacobians
     * between the two sensitives multiplicatively. To this end, we ask the
     * parameter transporter to multiply the Jacobians into this matrix, which
     * is set to the multiplicative identity.
     */
    if (cfg.run_mbf_smoother) {
        assert(payload.tmp_jacobian_ptr != nullptr);

        payload.tmp_jacobian_ptr[param_id] = matrix::identity<
            bound_matrix<typename propagator_t::detector_type::algebra_type>>();
        // s1._full_jacobian_ptr = &payload.tmp_jacobian_ptr[param_id];
    }

    s5.min_pT(static_cast<scalar_t>(cfg.min_pT));
    s5.min_p(static_cast<scalar_t>(cfg.min_p));
    // NOTE: CKF aborter is now s7, instead of s6
    s7.min_step_length = cfg.min_step_length_for_next_surface;
    s7.max_count = cfg.max_step_counts_for_next_surface;

    // Propagate to the next surface
    //NOTE: transporter not in device actor chain now, so not included in tie
    // propagator.propagate(propagation, detray::tie(s0, s1, s2, s3, s4, s5, s6));
    // Run a perigee prepass for step 0, initialize pattern, wire collector, 
    // then propagate with new actor chain.
    if (expected_layer_patterns.size() > link_idx) {
        expected_layer_pattern_type perigee_pattern{};
        bool perigee_pattern_ready = false;

        if (payload.step == 0 && payload.expected_layer_map != nullptr &&
            payload.expected_layer_map_size > 0u) {
            using detector_t = typename propagator_t::detector_type;
            using algebra_t = typename detector_t::algebra_type;
            using scalar_t = typename detector_t::scalar_type;
            using nav_link_t =
                typename detector_t::surface_type::navigation_link;
            using perigee_surface_t =
                detray::external_surface<algebra_t, detray::line_circular,
                                         detray::material_rod<scalar_t>,
                                         nav_link_t>;
            using perigee_navigator_t =
                detray::direct_navigator<
                    detector_t,
                    detray::ranges::single_view<perigee_surface_t>>;
            using perigee_propagator_t =
                detray::propagator<typename propagator_t::stepper_type,
                                   perigee_navigator_t, detray::actor_chain<>>;

            static constexpr typename perigee_surface_t::material_type
                perigee_material{detray::vacuum<scalar_t>{},
                                 std::numeric_limits<scalar_t>::max()};
            static constexpr typename perigee_surface_t::mask_type perigee_mask{
                0u, std::numeric_limits<scalar_t>::max(),
                -std::numeric_limits<scalar_t>::max()};
            perigee_surface_t perigee_surface{
                detray::dtransform3D<algebra_t>{}, perigee_mask,
                perigee_material, 0u, detray::surface_id::e_passive};

            const detray::tracking_surface sf{det, in_par.surface_link()};
            const auto free_seed =
                sf.bound_to_free_vector(typename detector_t::geometry_context{},
                                        in_par);

            perigee_propagator_t perigee_propagator(prop_cfg);
            typename perigee_propagator_t::state perigee_state(
                free_seed, payload.field_data, det,
                detray::ranges::single_view<perigee_surface_t>{perigee_surface},
                prop_cfg.context);
            perigee_state.set_particle(detail::correct_particle_hypothesis(
                cfg.ptc_hypothesis, free_seed));
            perigee_state._stepping
                .template set_constraint<detray::step::constraint::e_accuracy>(
                    cfg.propagation.stepping.step_constraint);
            perigee_state._navigation.set_direction(
                detray::navigation::direction::e_backward);
            perigee_propagator.propagate(perigee_state);

            if (perigee_propagator.finished(perigee_state) &&
                !perigee_state._stepping().is_invalid()) {
                propagator_t perigee_to_first(prop_cfg);
                typename propagator_t::state perigee_propagation(
                    perigee_state._stepping(), payload.field_data, det);
                perigee_propagation.set_particle(
                    detail::correct_particle_hypothesis(
                        cfg.ptc_hypothesis, perigee_state._stepping()));
                perigee_propagation._stepping
                    .template set_constraint<
                        detray::step::constraint::e_accuracy>(
                        cfg.propagation.stepping.step_constraint);

                typename detray::detail::tuple_element<0, actor_tuple_type>::type::
                    state p0{};
                typename detray::detail::tuple_element<3, actor_tuple_type>::type::
                    state p3{};
                typename detray::detail::tuple_element<2, actor_tuple_type>::type::
                    state p2{p3};
                typename detray::detail::tuple_element<4, actor_tuple_type>::type::
                    state p4{prop_cfg};
                typename detray::detail::tuple_element<5, actor_tuple_type>::type::
                    state p5;
                typename detray::detail::tuple_element<6, actor_tuple_type>::type::
                    state p6;
                typename detray::detail::tuple_element<7, actor_tuple_type>::type::
                    state p7;

                p5.min_pT(static_cast<scalar_t>(cfg.min_pT));
                p5.min_p(static_cast<scalar_t>(cfg.min_p));
                p7.min_step_length = cfg.min_step_length_for_next_surface;
                p7.max_count = cfg.max_step_counts_for_next_surface;

                p6.pattern = &perigee_pattern;
                p6.mapper.entries = payload.expected_layer_map;
                p6.mapper.size = payload.expected_layer_map_size;

                perigee_to_first.propagate(
                    perigee_propagation,
                    detray::tie(p0, p2, p3, p4, p5, p6, p7));

                perigee_pattern_ready = p7.success;
                if (globalIndex < 3u) {
                    TRACCC_INFO_DEVICE(
                        "Perigee prepass idx=%u success=%u updates=%u "
                        "skipped=%u",
                        globalIndex, static_cast<unsigned int>(p7.success),
                        p6.n_updates, p6.n_skipped);
                }
            } else if (globalIndex < 3u) {
                TRACCC_INFO_DEVICE(
                    "Perigee prepass idx=%u failed perigee_extrapolation",
                    globalIndex);
            }
        }

        //NOTE: ADDED TO FIX GARBAGE VALUE ISSUE IN THE EXPECTED LAYER PATTERN
        auto& link_pattern = expected_layer_patterns.at(link_idx);
        for (unsigned int i = 0u; i < link_pattern.size(); ++i) {
            link_pattern[i] = 0u;
        }
        if (perigee_pattern_ready) {
            for (unsigned int i = 0u; i < link_pattern.size(); ++i) {
                link_pattern[i] = perigee_pattern[i];
            }
        }
        s6.pattern = &link_pattern;
        s6.mapper.entries = payload.expected_layer_map;
        s6.mapper.size = payload.expected_layer_map_size;
    }

    propagator.propagate(propagation,
                         detray::tie(s0, s2, s3, s4, s5, s6, s7));

    // If a surface found, add the parameter for the next step
    // NOTE: CKF aborter is now s7, instead of s6
    if (s7.success) {
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
    }
}

}  // namespace traccc::device
