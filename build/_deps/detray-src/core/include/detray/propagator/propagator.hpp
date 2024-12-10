/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/macros.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/navigation/intersection/intersection.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/base_stepper.hpp"
#include "detray/propagator/propagation_config.hpp"
#include "detray/tracks/tracks.hpp"

// System include(s).
#include <iomanip>

namespace detray {

/// Templated propagator class, using a stepper and a navigator object in
/// succession.
///
/// @tparam stepper_t for the transport
/// @tparam navigator_t for the navigation
template <typename stepper_t, typename navigator_t, typename actor_chain_t>
struct propagator {

    using stepper_type = stepper_t;
    using navigator_type = navigator_t;
    using intersection_type = typename navigator_type::intersection_type;
    using detector_type = typename navigator_type::detector_type;
    using actor_chain_type = actor_chain_t;
    using algebra_type = typename stepper_t::algebra_type;
    using scalar_type = dscalar<algebra_type>;
    using free_track_parameters_type =
        typename stepper_t::free_track_parameters_type;
    using bound_track_parameters_type =
        typename stepper_t::bound_track_parameters_type;

    propagation::config m_cfg;

    stepper_t m_stepper;
    navigator_t m_navigator;

    /// Register the actor types
    const actor_chain_t run_actors{};

    /// Construct from a propagator configuration
    DETRAY_HOST_DEVICE
    explicit constexpr propagator(const propagation::config &cfg)
        : m_cfg{cfg} {}

    /// Propagation that state aggregates a stepping and a navigation state. It
    /// also keeps references to the actor states.
    struct state {

        using detector_type = typename navigator_t::detector_type;
        using context_type = typename detector_type::geometry_context;
        using navigator_state_type = typename navigator_t::state;
        using actor_chain_type = actor_chain_t;
        using scalar_type = typename navigator_t::scalar_type;

        /// Construct the propagation state with free parameter
        DETRAY_HOST_DEVICE state(const free_track_parameters_type &free_params,
                                 const detector_type &det,
                                 const context_type &ctx)
            : _stepping(free_params), _navigation(det), _context(ctx) {}

        /// Construct the propagation state with free parameter
        template <typename field_t>
        DETRAY_HOST_DEVICE state(const free_track_parameters_type &free_params,
                                 const field_t &magnetic_field,
                                 const detector_type &det,
                                 const context_type &ctx = {})
            : _stepping(free_params, magnetic_field),
              _navigation(det),
              _context(ctx) {}

        /// Construct the propagation state from the navigator state view
        DETRAY_HOST_DEVICE state(
            const free_track_parameters_type &free_params,
            const detector_type &det,
            typename navigator_type::state::view_type nav_view,
            const context_type &ctx = {})
            : _stepping(free_params),
              _navigation(det, nav_view),
              _context(ctx) {}

        /// Construct the propagation state from the navigator state view
        template <typename field_t>
        DETRAY_HOST_DEVICE state(
            const free_track_parameters_type &free_params,
            const field_t &magnetic_field, const detector_type &det,
            typename navigator_type::state::view_type nav_view,
            const context_type &ctx = {})
            : _stepping(free_params, magnetic_field),
              _navigation(det, nav_view),
              _context(ctx) {}

        /// Construct the propagation state with bound parameter
        DETRAY_HOST_DEVICE state(const bound_track_parameters_type &param,
                                 const detector_type &det,
                                 const context_type &ctx = {})
            : _stepping(param, det, ctx), _navigation(det), _context(ctx) {}

        /// Construct the propagation state with bound parameter
        template <typename field_t>
        DETRAY_HOST_DEVICE state(const bound_track_parameters_type &param,
                                 const field_t &magnetic_field,
                                 const detector_type &det,
                                 const context_type &ctx = {})
            : _stepping(param, magnetic_field, det, ctx),
              _navigation(det),
              _context(ctx) {}

        /// Set the particle hypothesis
        DETRAY_HOST_DEVICE
        void set_particle(const pdg_particle<scalar_type> &ptc) {
            _stepping.set_particle(ptc);
        }

        /// @returns the propagation heartbeat
        DETRAY_HOST_DEVICE
        bool is_alive() const { return _heartbeat; }

        // Is the propagation still alive?
        bool _heartbeat = false;

        typename stepper_t::state _stepping;
        typename navigator_t::state _navigation;
        context_type _context;

        bool do_debug = false;
#if defined(__NO_DEVICE__)
        std::stringstream debug_stream{};
#endif
    };

    /// Propagate method init: Initialize a propagation state
    ///
    /// @param propagation the state of a propagation flow
    /// @param actor_state_refs tuple containing refences to the actor states
    ///
    /// @return The heartbeat at the end of the initialization.
    ///
    /// @note If the return value of this function is true, a propagation step
    /// can be taken afterwards.
    DETRAY_HOST_DEVICE void propagate_init(
        state &propagation,
        typename actor_chain_t::state actor_state_refs) const {
        auto &navigation = propagation._navigation;
        auto &stepping = propagation._stepping;
        auto &context = propagation._context;
        const auto &track = stepping();

        // Initialize the navigation
        m_navigator.init(track, navigation, m_cfg.navigation, context);
        propagation._heartbeat = navigation.is_alive();

        // Run all registered actors/aborters after init
        run_actors(actor_state_refs, propagation);

        // Find next candidate
        m_navigator.update(track, navigation, m_cfg.navigation, context);
        propagation._heartbeat &= navigation.is_alive();
    }

    /// Propagate method step: Perform a single propagation step.
    ///
    /// @param propagation the state of a propagation flow
    /// @param actor_state_refs tuple containing refences to the actor states
    ///
    /// @return The heartbeat at the end of the step.
    ///
    /// @note If the return value of this function is true, another step can
    /// be taken afterwards.
    DETRAY_HOST_DEVICE bool propagate_step(
        state &propagation, bool is_init,
        typename actor_chain_t::state actor_state_refs) const {
        auto &navigation = propagation._navigation;
        auto &stepping = propagation._stepping;
        auto &context = propagation._context;
        const auto &track = stepping();

        // Set access to the volume material for the stepper
        auto vol = navigation.get_volume();
        const material<scalar_type> *vol_mat_ptr =
            vol.has_material() ? vol.material_parameters(track.pos()) : nullptr;

        // Break automatic step size scaling by the stepper when a surface
        // was reached and whenever the navigation is (re-)initialized
        const bool reset_stepsize{navigation.is_on_surface() || is_init};
        // Take the step
        propagation._heartbeat &=
            m_stepper.step(navigation(), stepping, m_cfg.stepping,
                           reset_stepsize, vol_mat_ptr);

        // Reduce navigation trust level according to stepper update
        typename stepper_t::policy_type{}(stepping.policy_state(), propagation);

        // Find next candidate
        is_init =
            m_navigator.update(track, navigation, m_cfg.navigation, context);
        propagation._heartbeat &= navigation.is_alive();

        // Run all registered actors/aborters after update
        run_actors(actor_state_refs, propagation);

        // And check the status
        is_init |=
            m_navigator.update(track, navigation, m_cfg.navigation, context);
        propagation._heartbeat &= navigation.is_alive();

#if defined(__NO_DEVICE__)
        if (propagation.do_debug) {
            inspect(propagation);
        }
#endif

        return is_init;
    }

    /// Propagate method finale: Return whether or not the propagation
    /// completed succesfully.
    ///
    /// @param propagation the state of a propagation flow
    ///
    /// @return propagation success.
    DETRAY_HOST_DEVICE bool propagate_is_complete(state &propagation) const {
        return propagation._navigation.is_complete();
    }

    /// Propagate method: Coordinates the calls of the stepper, navigator and
    /// all registered actors.
    ///
    /// @param propagation the state of a propagation flow
    /// @param actor_state_refs tuple containing refences to the actor states
    ///
    /// @return propagation success.
    DETRAY_HOST_DEVICE bool propagate(
        state &propagation,
        typename actor_chain_t::state actor_state_refs) const {

        propagate_init(propagation, actor_state_refs);
        bool is_init = true;

        // Run while there is a heartbeat
        while (propagation.is_alive()) {
            is_init = propagate_step(propagation, is_init, actor_state_refs);
        }

        // Pass on the whether the propagation was successful
        return propagate_is_complete(propagation);
    }

    /// Overload for emtpy actor chain
    DETRAY_HOST_DEVICE bool propagate(state &propagation) {
        // Will not be used
        actor_chain<>::state empty_state{};
        // Run propagation
        return propagate(propagation, empty_state);
    }

    /// Propagate method with two while loops. In the CPU, propagate and
    /// propagate_sync() should be equivalent to each other. In the SIMT level
    /// (e.g. GPU), the instruction of threads in the same warp is synchornized
    /// after the internal while loop.
    ///
    /// @tparam state_t is the propagation state type
    /// @tparam actor_state_t is the actor state type
    ///
    /// @param propagation the state of a propagation flow
    /// @param actor_states the actor state
    ///
    /// @return propagation success.
    DETRAY_HOST_DEVICE bool propagate_sync(
        state &propagation,
        typename actor_chain_t::state actor_state_refs) const {

        propagate_init(propagation, actor_state_refs);
        bool is_init = true;

        auto &navigation = propagation._navigation;
        auto &stepping = propagation._stepping;
        auto &context = propagation._context;
        const auto &track = stepping();

        while (propagation.is_alive()) {

            bool skip = true;

            while (propagation.is_alive()) {

                // Set access to the volume material for the stepper
                auto vol = navigation.get_volume();
                const material<scalar_type> *vol_mat_ptr =
                    vol.has_material() ? vol.material_parameters(track.pos())
                                       : nullptr;

                // Break automatic step size scaling by the stepper
                const bool reset_stepsize{navigation.is_on_surface() ||
                                          is_init};
                // Take the step
                propagation._heartbeat &=
                    m_stepper.step(navigation(), stepping, m_cfg.stepping,
                                   reset_stepsize, vol_mat_ptr);

                // Reduce navigation trust level according to stepper update
                typename stepper_t::policy_type{}(stepping.policy_state(),
                                                  propagation);

                // Find next candidate
                is_init = m_navigator.update(track, navigation,
                                             m_cfg.navigation, context);
                propagation._heartbeat &= navigation.is_alive();

                // If the track is on a sensitive surface, break the loop to
                // synchornize the threads
                if (propagation._navigation.is_on_sensitive()) {
                    skip = false;
                    break;
                } else {
                    run_actors(actor_state_refs, propagation);

                    // And check the status
                    is_init |= m_navigator.update(track, navigation,
                                                  m_cfg.navigation, context);
                    propagation._heartbeat &= navigation.is_alive();
                }
            }

            if (!skip) {

                // Synchronized actor
                run_actors(actor_state_refs, propagation);

                // And check the status
                is_init |= m_navigator.update(track, navigation,
                                              m_cfg.navigation, context);
                propagation._heartbeat &= navigation.is_alive();
            }

#if defined(__NO_DEVICE__)
            if (propagation.do_debug) {
                inspect(propagation);
            }
#endif
        }

        // Pass on the whether the propagation was successful
        return propagate_is_complete(propagation);
    }

    template <typename state_t>
    DETRAY_HOST void inspect(state_t &propagation) const {
        const auto &navigation = propagation._navigation;
        const auto &stepping = propagation._stepping;

        propagation.debug_stream << std::left << std::setw(30);
        switch (navigation.status()) {
            using enum navigation::status;
            case e_abort:
                propagation.debug_stream << "status: abort";
                break;
            case e_on_target:
                propagation.debug_stream << "status: e_on_target";
                break;
            case e_unknown:
                propagation.debug_stream << "status: unknowm";
                break;
            case e_towards_object:
                propagation.debug_stream << "status: towards_surface";
                break;
            case e_on_module:
                propagation.debug_stream << "status: on_module";
                break;
            case e_on_portal:
                propagation.debug_stream << "status: on_portal";
                break;
            default:
                break;
        }

        if (detail::is_invalid_value(navigation.volume())) {
            propagation.debug_stream << "volume: " << std::setw(10)
                                     << "invalid";
        } else {
            propagation.debug_stream << "volume: " << std::setw(10)
                                     << navigation.volume();
        }

        propagation.debug_stream << "surface: " << std::setw(14);
        if (navigation.is_on_surface()) {
            propagation.debug_stream << navigation.barcode();
        } else {
            propagation.debug_stream << "undefined";
        }

        propagation.debug_stream << "step_size: " << std::setw(10)
                                 << stepping.step_size() << std::endl;

        propagation.debug_stream << std::setw(10)
                                 << detail::ray<algebra_type>(stepping())
                                 << std::endl;
    }
};

}  // namespace detray
