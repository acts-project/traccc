/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/units.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/tracks/tracks.hpp"

// Detray test include(s)
#include "detray/test/utils/inspectors.hpp"
#include "detray/test/utils/simulation/event_generator/track_generators.hpp"
#include "detray/test/validation/step_tracer.hpp"

// Vecmem include(s)
#include <vecmem/memory/memory_resource.hpp>

// Covfie include(s)
#include <covfie/core/field.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System include(s)
#include <stdexcept>
#include <tuple>

namespace detray {

// These types are identical in host and device code for all bfield types
using algebra_t = ALGEBRA_PLUGIN<detray::scalar>;
using scalar_t = dscalar<algebra_t>;
using vector3_t = dvector3D<algebra_t>;
using point3_t = dpoint3D<algebra_t>;
using matrix_operator = dmatrix_operator<algebra_t>;
using track_t = free_track_parameters<algebra_t>;
using free_matrix_t = free_matrix<algebra_t>;

constexpr std::size_t cache_size{navigation::default_cache_size};

// Navigator
template <typename detector_t, typename inspector_t>
using navigator_w_insp_t = navigator<detector_t, cache_size, inspector_t>;
template <typename detector_t>
using navigator_t = navigator_w_insp_t<detector_t, navigation::void_inspector>;
template <typename detector_t>
using intersection_t = typename navigator_t<detector_t>::intersection_type;

// Stepper
using constraints_t = constrained_step<>;
template <typename bfield_view_t>
using rk_stepper_t = rk_stepper<bfield_view_t, algebra_t, constraints_t>;

// Track generator
using generator_t = uniform_track_generator<track_t>;

/// Test tolerance
constexpr scalar_t is_close{1e-4f};

/// Test configuration
struct propagator_test_config {
    generator_t::configuration track_generator;
    propagation::config propagation;
};

// Assemble actor chain type
using step_tracer_host_t = step_tracer<algebra_t, vecmem::vector>;
using step_tracer_device_t = step_tracer<algebra_t, vecmem::device_vector>;
using actor_chain_host_t =
    actor_chain<tuple, step_tracer_host_t, pathlimit_aborter,
                parameter_transporter<algebra_t>,
                pointwise_material_interactor<algebra_t>,
                parameter_resetter<algebra_t>>;
using actor_chain_device_t =
    actor_chain<tuple, step_tracer_device_t, pathlimit_aborter,
                parameter_transporter<algebra_t>,
                pointwise_material_interactor<algebra_t>,
                parameter_resetter<algebra_t>>;

/// Precompute the tracks
template <typename track_generator_t = uniform_track_generator<track_t>>
inline auto generate_tracks(
    vecmem::memory_resource *mr,
    const typename track_generator_t::configuration &cfg = {}) {

    // Track collection
    vecmem::vector<typename track_generator_t::track_type> tracks(mr);

    // Iterate through uniformly distributed momentum directions
    for (auto track : track_generator_t{cfg}) {
        // Put it into vector of trajectories
        tracks.push_back(track);
    }

    return tracks;
}

/// Test function for propagator on the host
template <typename bfield_bknd_t, typename host_detector_t>
inline auto run_propagation_host(vecmem::memory_resource *mr,
                                 const host_detector_t &det,
                                 const propagation::config &cfg,
                                 covfie::field<bfield_bknd_t> &field,
                                 const vecmem::vector<track_t> &tracks)
    -> vecmem::jagged_vector<detail::step_data<algebra_t>> {

    // Construct propagator from stepper and navigator
    using host_stepper_t =
        rk_stepper_t<typename covfie::field<bfield_bknd_t>::view_t>;
    using host_navigator_t =
        navigator_w_insp_t<host_detector_t, navigation::print_inspector>;
    using propagator_host_t =
        propagator<host_stepper_t, host_navigator_t, actor_chain_host_t>;

    propagator_host_t p{cfg};

    // Create vector for track recording
    vecmem::jagged_vector<detail::step_data<algebra_t>> host_steps(mr);

    for (const auto &trk : tracks) {

        // Create the propagator state
        step_tracer_host_t::state tracer_state{*mr};
        tracer_state.collect_only_on_surface(true);
        pathlimit_aborter::state pathlimit_state{cfg.stepping.path_limit};
        parameter_transporter<algebra_t>::state transporter_state{};
        pointwise_material_interactor<algebra_t>::state interactor_state{};
        parameter_resetter<algebra_t>::state resetter_state{};
        auto actor_states =
            detray::tie(tracer_state, pathlimit_state, transporter_state,
                        interactor_state, resetter_state);

        typename propagator_host_t::state state(trk, field, det);

        state._stepping.template set_constraint<step::constraint::e_accuracy>(
            cfg.stepping.step_constraint);

        // Run propagation
        if (!p.propagate(state, actor_states)) {
            std::cout << state._navigation.inspector().to_string() << std::endl;
            throw std::runtime_error("Host propagation failed");
        } else if (tracer_state.get_step_data().empty()) {
            std::cout << state._navigation.inspector().to_string() << std::endl;
            throw std::runtime_error(
                "Host propagation did not record reference data correctly");
        }

        host_steps.push_back(std::move(tracer_state.get_step_data()));
    }

    return host_steps;
}

/// Compare the results between host and device propagation
inline void compare_propagation_results(
    const vecmem::jagged_vector<detail::step_data<algebra_t>> &host_steps,
    const vecmem::jagged_vector<detail::step_data<algebra_t>> &device_steps) {

    // Make sure the same number of tracks were tested on both backends
    ASSERT_EQ(host_steps.size(), device_steps.size());

    // Compare the positions and pathlengths
    for (unsigned int i = 0u; i < host_steps.size(); i++) {

        EXPECT_TRUE(host_steps[i].size() > 0u)
            << "No surfaces forund on track " << i;

        // Recorded as many positions as path lengths
        EXPECT_EQ(host_steps[i].size(), device_steps[i].size());

        for (unsigned int j = 0u;
             j < math::min(host_steps[i].size(), device_steps[i].size()); j++) {

            // Compare recorded path lengths along track
            const auto &host_step = host_steps[i][j];
            const auto &device_step = device_steps[i][j];

            EXPECT_NEAR(host_step.path_length, device_step.path_length,
                        host_step.path_length * is_close)
                << "ERROR: Path length at track " << i << " step " << j
                << std::endl;

            // Compare recorded positions along track
            const point3_t &host_pos = host_step.track_params.pos();
            const point3_t &device_pos = device_step.track_params.pos();

            auto relative_error = static_cast<point3_t>(
                1.f / host_step.path_length * (host_pos - device_pos));

            EXPECT_NEAR(getter::norm(relative_error), 0.f, is_close)
                << "ERROR: Position at track " << i << " step " << j << ": ["
                << host_pos[0] << ", " << host_pos[1] << ", " << host_pos[2]
                << "] (host), [" << device_pos[0] << ", " << device_pos[1]
                << ", " << device_pos[2] << "] (device)" << std::endl;

            // Compare the Jacobians
            const free_matrix_t &host_J = host_step.jacobian;
            const free_matrix_t &device_J = device_step.jacobian;

            for (std::size_t row = 0u; row < e_free_size; row++) {
                for (std::size_t col = 0u; col < e_free_size; col++) {

                    scalar_t host_val =
                        matrix_operator().element(host_J, row, col);

                    scalar_t device_val =
                        matrix_operator().element(device_J, row, col);

                    ASSERT_NEAR((host_val - device_val) / host_step.path_length,
                                0.f, is_close)
                        << "ERROR: matrix element mismatch at row " << row
                        << ", col " << col << std::endl;
                }
            }
        }
    }
}

}  // namespace detray
