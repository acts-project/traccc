/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/definitions/units.hpp"
#include "detray/materials/interaction.hpp"
#include "detray/materials/predefined_materials.hpp"
#include "detray/navigation/policies.hpp"
#include "detray/propagator/base_stepper.hpp"
#include "detray/tracks/tracks.hpp"
#include "detray/utils/matrix_helper.hpp"

namespace detray {

/// Runge-Kutta-Nystrom 4th order stepper implementation
///
/// @tparam magnetic_field_t the type of magnetic field
/// @tparam track_t the type of track that is being advanced by the stepper
/// @tparam constraint_ the type of constraints on the stepper
template <typename magnetic_field_t, typename algebra_t,
          typename constraint_t = unconstrained_step,
          typename policy_t = stepper_rk_policy,
          typename inspector_t = stepping::void_inspector>
class rk_stepper final
    : public base_stepper<algebra_t, constraint_t, policy_t, inspector_t> {

    using base_type =
        base_stepper<algebra_t, constraint_t, policy_t, inspector_t>;

    public:
    using algebra_type = algebra_t;
    using scalar_type = dscalar<algebra_t>;
    using point3_type = dpoint3D<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using transform3_type = dtransform3D<algebra_t>;
    using matrix_operator = dmatrix_operator<algebra_t>;
    using free_track_parameters_type =
        typename base_type::free_track_parameters_type;
    using bound_track_parameters_type =
        typename base_type::bound_track_parameters_type;
    using magnetic_field_type = magnetic_field_t;
    template <std::size_t ROWS, std::size_t COLS>
    using matrix_type = dmatrix<algebra_t, ROWS, COLS>;

    rk_stepper() = default;

    struct intermediate_state {
        vector3_type b_first{0.f, 0.f, 0.f};
        vector3_type b_middle{0.f, 0.f, 0.f};
        vector3_type b_last{0.f, 0.f, 0.f};
        // t = tangential direction = dr/ds
        std::array<vector3_type, 4u> t;
        // q/p
        std::array<scalar_type, 4u> qop;
        // dt/ds = d^2r/ds^2 = q/p ( t X B )
        std::array<vector3_type, 4u> dtds;
        // d(q/p)/ds
        std::array<scalar_type, 4u> dqopds;
    };

    struct state : public base_type::state {

        friend rk_stepper;

        static constexpr const stepping::id id = stepping::id::e_rk;

        DETRAY_HOST_DEVICE
        state(const free_track_parameters_type& t,
              const magnetic_field_t& mag_field)
            : base_type::state(t), m_magnetic_field(mag_field) {}

        template <typename detector_t>
        DETRAY_HOST_DEVICE state(
            const bound_track_parameters_type& bound_params,
            const magnetic_field_t& mag_field, const detector_t& det,
            const typename detector_t::geometry_context& ctx)
            : base_type::state(bound_params, det, ctx),
              m_magnetic_field(mag_field) {}

        /// @returns the B-field view
        magnetic_field_type field() const { return m_magnetic_field; }

        /// Set the next step size
        DETRAY_HOST_DEVICE
        inline void set_next_step_size(const scalar_type step) {
            m_next_step_size = step;
        }

        /// @returns the next step size to be taken on the following step.
        DETRAY_HOST_DEVICE
        inline scalar_type next_step_size() const { return m_next_step_size; }

        /// Update the track state by Runge-Kutta-Nystrom integration.
        DETRAY_HOST_DEVICE
        void advance_track(const intermediate_state& sd,
                           const material<scalar_type>* vol_mat_ptr);

        /// Update the jacobian transport from free propagation
        DETRAY_HOST_DEVICE
        void advance_jacobian(const stepping::config& cfg,
                              const intermediate_state&,
                              const material<scalar_type>* vol_mat_ptr);

        /// evaulate dqopds for a given step size and material
        DETRAY_HOST_DEVICE
        detray::pair<scalar_type, scalar_type> evaluate_dqopds(
            const std::size_t i, const scalar_type h,
            const scalar_type dqopds_prev,
            const material<scalar_type>* vol_mat_ptr,
            const detray::stepping::config& cfg);

        /// evaulate dtds for runge kutta stepping
        DETRAY_HOST_DEVICE
        detray::pair<vector3_type, vector3_type> evaluate_dtds(
            const vector3_type& b_field, const std::size_t i,
            const scalar_type h, const vector3_type& dtds_prev,
            const scalar_type qop);

        DETRAY_HOST_DEVICE
        matrix_type<3, 3> evaluate_field_gradient(const point3_type& pos);

        /// Evaluate dtds, where t is the unit tangential direction
        DETRAY_HOST_DEVICE
        vector3_type dtds() const;

        /// Evaulate d(qop)/ds
        DETRAY_HOST_DEVICE
        scalar_type dqopds(const material<scalar_type>* vol_mat_ptr) const;

        DETRAY_HOST_DEVICE
        scalar_type dqopds(const scalar_type qop,
                           const material<scalar_type>* vol_mat_ptr) const;

        /// Evaulate d(d(qop)/ds)dqop
        DETRAY_HOST_DEVICE
        scalar_type d2qopdsdqop(const scalar_type qop,
                                const material<scalar_type>* vol_mat_ptr) const;

        /// Call the stepping inspector
        template <typename... Args>
        DETRAY_HOST_DEVICE void run_inspector(
            [[maybe_unused]] const stepping::config& cfg,
            [[maybe_unused]] const char* message,
            [[maybe_unused]] Args&&... args) {
            if constexpr (!std::is_same_v<inspector_t,
                                          stepping::void_inspector>) {
                this->inspector()(*this, cfg, message,
                                  std::forward<Args>(args)...);
            }
        }

        private:
        vector3_type m_dtds_3;
        scalar_type m_dqopds_3;

        /// Next step size after adaptive step size scaling
        scalar_type m_next_step_size{0.f};

        /// Magnetic field view
        const magnetic_field_t m_magnetic_field;
    };

    /// Take a step, using an adaptive Runge-Kutta algorithm.
    ///
    /// @param dist_to_next The straight line distance to the next surface
    /// @param stepping The state object of a stepper
    /// @param cfg The stepping configuration
    /// @param do_reset whether to reset the RKN step size to "dist to next"
    ///
    /// @return returning the heartbeat, indicating if the stepping is alive
    DETRAY_HOST_DEVICE bool step(
        const scalar_type dist_to_next, state& stepping,
        const stepping::config& cfg, bool do_reset,
        const material<scalar_type>* vol_mat_ptr = nullptr) const;
};

}  // namespace detray

#include "detray/propagator/rk_stepper.ipp"
