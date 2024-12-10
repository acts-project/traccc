/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/pdg_particle.hpp"
#include "detray/definitions/units.hpp"
#include "detray/geometry/barcode.hpp"
#include "detray/geometry/tracking_surface.hpp"
#include "detray/materials/material.hpp"
#include "detray/propagator/constrained_step.hpp"
#include "detray/propagator/stepping_config.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/curvilinear_frame.hpp"

namespace detray {

namespace stepping {

/// A void inpector that does nothing.
///
/// Inspectors can be plugged in to understand the current stepper state.
struct void_inspector {
    template <typename state_t>
    DETRAY_HOST_DEVICE constexpr void operator()(
        const state_t & /*ignored*/, const char * /*ignored*/) const {
        /*Do nothing*/
    }
};

}  // namespace stepping

/// Base stepper implementation
template <typename algebra_t, typename constraint_t, typename policy_t,
          typename inspector_t = stepping::void_inspector>
class base_stepper {

    public:
    using scalar_type = dscalar<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;
    using free_track_parameters_type = free_track_parameters<algebra_t>;
    using bound_track_parameters_type = bound_track_parameters<algebra_t>;
    using free_matrix_type = free_matrix<algebra_t>;
    using bound_matrix_type = bound_matrix<algebra_t>;

    using inspector_type = inspector_t;
    using policy_type = policy_t;

    /// @brief State struct holding the track
    ///
    /// @note It has to cast into a const track via the call operation.
    struct state {

        /// Sets track parameters.
        DETRAY_HOST_DEVICE
        explicit state(const free_track_parameters_type &free_params)
            : m_track(free_params) {

            curvilinear_frame<algebra_t> cf(free_params);

            // Set bound track parameters
            m_bound_params.set_parameter_vector(cf.m_bound_vec);

            // A dummy covariance - should not be used
            m_bound_params.set_covariance(
                matrix_operator()
                    .template identity<e_bound_size, e_bound_size>());

            // An invalid barcode - should not be used
            m_bound_params.set_surface_link(geometry::barcode{});
        }

        /// Sets track parameters from bound track parameter.
        template <typename detector_t>
        DETRAY_HOST_DEVICE state(
            const bound_track_parameters_type &bound_params,
            const detector_t &det,
            const typename detector_t::geometry_context &ctx)
            : m_bound_params(bound_params) {

            // Departure surface
            const auto sf = tracking_surface{det, bound_params.surface_link()};

            // Set free track parameters for stepping/navigation
            m_track = sf.bound_to_free_vector(ctx, bound_params);
        }

        /// @returns free track parameters - non-const access
        DETRAY_HOST_DEVICE
        free_track_parameters_type &operator()() { return m_track; }

        /// @returns free track parameters - const access
        DETRAY_HOST_DEVICE
        const free_track_parameters_type &operator()() const { return m_track; }

        /// @returns bound track parameters - const access
        DETRAY_HOST_DEVICE
        bound_track_parameters_type &bound_params() { return m_bound_params; }

        /// @returns bound track parameters - non-const access
        DETRAY_HOST_DEVICE
        const bound_track_parameters_type &bound_params() const {
            return m_bound_params;
        }

        /// Get stepping direction
        DETRAY_HOST_DEVICE
        inline step::direction direction() const {
            return m_step_size >= 0.f ? step::direction::e_forward
                                      : step::direction::e_backward;
        }

        /// Updates the total number of step trials
        DETRAY_HOST_DEVICE inline void count_trials() { ++m_n_total_trials; }

        /// @returns the total number of step trials. For steppers that don't
        /// use adaptive step size scaling, this is the number of steps
        DETRAY_HOST_DEVICE inline std::size_t n_total_trials() const {
            return m_n_total_trials;
        }

        /// Set the current step size
        DETRAY_HOST_DEVICE
        inline void set_step_size(const scalar_type step) {
            m_step_size = step;
        }

        /// @returns the step size of the current step.
        DETRAY_HOST_DEVICE
        inline scalar_type step_size() const { return m_step_size; }

        /// @returns this states path length.
        DETRAY_HOST_DEVICE
        inline scalar_type path_length() const { return m_path_length; }

        /// @returns this states total integration path length.
        DETRAY_HOST_DEVICE
        inline scalar_type abs_path_length() const { return m_abs_path_length; }

        /// Add a new segment to all path lengths (forward or backward)
        DETRAY_HOST_DEVICE inline void update_path_lengths(scalar_type seg) {
            m_path_length += seg;
            m_abs_path_length += math::fabs(seg);
        }

        /// Set new step constraint
        template <step::constraint type = step::constraint::e_actor>
        DETRAY_HOST_DEVICE inline void set_constraint(scalar_type step_size) {
            m_constraint.template set<type>(step_size);
        }

        /// @returns access to this states step constraints
        DETRAY_HOST_DEVICE
        inline const constraint_t &constraints() const { return m_constraint; }

        /// Remove [all] constraints
        template <step::constraint type = step::constraint::e_actor>
        DETRAY_HOST_DEVICE inline void release_step() {
            m_constraint.template release<type>();
        }

        /// @returns the current particle hypothesis
        DETRAY_HOST_DEVICE
        const auto &particle_hypothesis() const { return m_ptc; }

        /// Set new particle hypothesis
        DETRAY_HOST_DEVICE
        inline void set_particle(const pdg_particle<scalar_type> &ptc) {
            m_ptc = ptc;
        }

        /// @returns the current transport Jacbian.
        DETRAY_HOST_DEVICE
        inline const free_matrix_type &transport_jacobian() const {
            return m_jac_transport;
        }

        /// @returns the current full Jacbian.
        DETRAY_HOST_DEVICE
        inline const bound_matrix_type &full_jacobian() const {
            return m_full_jacobian;
        }

        /// Set new full Jacbian.
        DETRAY_HOST_DEVICE
        inline void set_full_jacobian(const bound_matrix_type &jac) {
            m_full_jacobian = jac;
        }

        /// Reset transport Jacbian.
        DETRAY_HOST_DEVICE
        inline void reset_transport_jacobian() {
            matrix_operator().set_identity(m_jac_transport);
        }

        /// @returns access to this states navigation policy state
        DETRAY_HOST_DEVICE
        inline typename policy_t::state &policy_state() {
            return m_policy_state;
        }

        /// @returns the stepping inspector
        DETRAY_HOST
        constexpr auto &inspector() { return m_inspector; }

        /// Call the stepping inspector
        DETRAY_HOST_DEVICE
        inline void run_inspector([[maybe_unused]] const stepping::config &cfg,
                                  [[maybe_unused]] const char *message) {
            if constexpr (!std::is_same_v<inspector_t,
                                          stepping::void_inspector>) {
                m_inspector(*this, cfg, message);
            }
        }

        protected:
        /// Set new transport Jacbian.
        DETRAY_HOST_DEVICE
        inline void set_transport_jacobian(const free_matrix_type &jac) {
            m_jac_transport = jac;
        }

        private:
        /// Jacobian transport matrix
        free_matrix_type m_jac_transport =
            matrix_operator().template identity<e_free_size, e_free_size>();

        /// Full jacobian
        bound_matrix_type m_full_jacobian =
            matrix_operator().template identity<e_bound_size, e_bound_size>();

        /// Bound covariance
        bound_track_parameters_type m_bound_params;

        /// Free track parameters
        free_track_parameters_type m_track;

        /// The default particle hypothesis is 'muon'
        pdg_particle<scalar_type> m_ptc = muon<scalar_type>();

        /// Total number of step trials
        std::size_t m_n_total_trials{0u};

        /// Current step size
        scalar_type m_step_size{0.f};

        /// Track path length (current position along track)
        scalar_type m_path_length{0.f};

        /// Absolute path length (total path length covered by the integration)
        scalar_type m_abs_path_length{0.f};

        /// Step size constraints (optional)
        [[no_unique_address]] constraint_t m_constraint = {};

        /// Navigation policy state
        [[no_unique_address]] typename policy_t::state m_policy_state = {};

        /// The inspector type of the stepping (for debugging only)
        [[no_unique_address]] inspector_type m_inspector;
    };
};

}  // namespace detray
