/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "detray/definitions/detail/algebra.hpp"
#include "detray/definitions/detail/qualifiers.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/tracks/free_track_parameters.hpp"

namespace detray {

namespace detail {
/// Data for a single step
template <typename algebra_t>
struct step_data {
    using scalar_type = dscalar<algebra_t>;
    using vector3_type = dvector3D<algebra_t>;
    using track_param_type = free_track_parameters<algebra_t>;
    using free_matrix_type = free_matrix<algebra_t>;

    scalar_type step_size{0.f};
    scalar_type path_length{0.f};
    std::size_t n_total_trials{0u};
    navigation::direction nav_dir = navigation::direction::e_forward;
    geometry::barcode barcode{};
    track_param_type track_params{};
    free_matrix_type jacobian{};
};
}  // namespace detail

/// Collect information at every step
template <typename algebra_t, template <typename...> class vector_t>
struct step_tracer : actor {

    using step_data_t = detail::step_data<algebra_t>;

    /// Actor state that collects the data
    struct state {
        friend struct step_tracer;

        state() = delete;

        /// Construct the vector containers with a given resource
        /// @param resource
        DETRAY_HOST
        explicit state(vecmem::memory_resource& resource)
            : m_steps(&resource) {}

        /// Construct from externally provided vector for the @param steps
        DETRAY_HOST_DEVICE
        explicit state(vector_t<step_data_t>&& steps)
            : m_steps(std::move(steps)) {}

        /// Access to the recorded step data of every step along the track -
        /// const
        DETRAY_HOST_DEVICE
        const auto& get_step_data() const { return m_steps; }

        /// Move the recorded step data out of the actor
        DETRAY_HOST
        auto&& release_step_data() && { return std::move(m_steps); }

        /// Collect the data at every step
        DETRAY_HOST_DEVICE
        void collect_every_step(bool do_collect_every_step = true) {
            m_collect_every_step = do_collect_every_step;
        }

        /// Collect the data only when on surface
        DETRAY_HOST_DEVICE
        void collect_only_on_surface(bool do_collect_every_step = true) {
            m_collect_every_step = !do_collect_every_step;
        }

        private:
        /// Whether to collect the step data at every step
        bool m_collect_every_step{true};
        /// The collected data for the steps
        vector_t<step_data_t> m_steps;
    };

    /// Actor call
    template <typename propagator_state_t>
    DETRAY_HOST_DEVICE void operator()(state& tracer_state,
                                       propagator_state_t& prop_state) const {
        const auto& navigation = prop_state._navigation;
        const auto& stepping = prop_state._stepping;

        // Collect the data whenever requested
        if (navigation.is_on_surface() || tracer_state.m_collect_every_step) {

            const geometry::barcode bcd{navigation.is_on_surface()
                                            ? navigation.barcode()
                                            : geometry::barcode{}};

            step_data_t sd{stepping.step_size(),
                           stepping.path_length(),
                           stepping.n_total_trials(),
                           navigation.direction(),
                           bcd,
                           stepping(),
                           stepping.transport_jacobian()};

            tracer_state.m_steps.push_back(std::move(sd));
        }
    }
};

}  // namespace detray
