/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/track_parameters.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/propagator/propagator.hpp"

// System include(s).
#include <random>

namespace traccc {

using matrix_operator = typename transform3::matrix_actor;

/// Seed track parameter generator
template <typename stepper_t, typename navigator_t>
struct seed_generator {

    /// Aborter actor which stops the propagation when the track reaches the
    /// first sensitive surface
    struct aborter : detray::actor {
        struct state {};

        /// Actor operation
        ///
        /// @param actor_state the actor state
        /// @param propagation the propagator state
        template <typename propagator_state_t>
        void operator()(state& /*actor_state*/,
                        propagator_state_t& propagation) const {

            auto& navigation = propagation._navigation;

            // Abort if the surface is sensitive
            if (navigation.is_on_sensitive()) {
                propagation._heartbeat &= navigation.abort();
            }
        }
    };

    /// Type declarations
    using transform3_type = typename stepper_t::transform3_type;
    using detector_type = typename navigator_t::detector_type;
    using transporter = detray::parameter_transporter<transform3_type>;
    using resetter = detray::parameter_resetter<transform3_type>;
    using actor_chain_type =
        detray::actor_chain<std::tuple, transporter, resetter, aborter>;
    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_chain_type>;

    /// Constructor with detector
    ///
    /// @param det input detector
    /// @param stddevs standard deviations for parameter smearing
    seed_generator(const detector_type& det,
                   const std::array<scalar, e_bound_size>& stddevs,
                   const std::size_t sd = 0)
        : m_detector(std::make_unique<detector_type>(det)), m_stddevs(stddevs) {
        generator.seed(sd);
    }

    /// Seed generator operation
    ///
    /// @param vertex vertex of particle
    /// @param stddevs standard deviations for track parameter smearing
    bound_track_parameters operator()(const free_track_parameters& vertex) {
        propagator_type propagator({}, {});
        auto actor_states =
            std::tie(m_transporter_state, m_resetter_state, m_aborter_state);
        typename propagator_type::state propagation(
            vertex, m_detector->get_bfield(), *m_detector);

        auto& stepping = propagation._stepping;

        propagator.propagate(propagation, actor_states);

        // Smeared vector and its covariance
        auto new_vec = matrix_operator().template zero<e_bound_size, 1>();
        auto new_cov =
            matrix_operator().template zero<e_bound_size, e_bound_size>();

        for (std::size_t i = 0; i < e_bound_size; i++) {

            matrix_operator().element(new_vec, i, 0) =
                std::normal_distribution<scalar>(
                    matrix_operator().element(stepping._bound_params.vector(),
                                              i, 0),
                    m_stddevs[i])(generator);

            matrix_operator().element(new_cov, i, i) =
                m_stddevs[i] * m_stddevs[i];
        }

        // Set vector and covariance
        stepping._bound_params.set_vector(new_vec);
        stepping._bound_params.set_covariance(new_cov);

        return stepping._bound_params;
    }

    private:
    // Random generator
    std::random_device rd{};
    std::mt19937 generator{rd()};

    /// Detector objects
    std::unique_ptr<detector_type> m_detector;
    /// Standard deviations for parameter smearing
    std::array<scalar, e_bound_size> m_stddevs;
    /// Actor states
    typename transporter::state m_transporter_state{};
    typename resetter::state m_resetter_state{};
    typename aborter::state m_aborter_state{};
};

}  // namespace traccc