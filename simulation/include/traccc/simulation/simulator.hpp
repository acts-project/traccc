/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/simulation/smearing_writer.hpp"
#include "traccc/utils/particle.hpp"

// Detray include(s).
#include <detray/definitions/pdg_particle.hpp>
#include <detray/navigation/navigator.hpp>
#include <detray/propagator/actor_chain.hpp>
#include <detray/propagator/actors/aborters.hpp>
#include <detray/propagator/actors/parameter_resetter.hpp>
#include <detray/propagator/actors/parameter_transporter.hpp>
#include <detray/propagator/propagator.hpp>
#include <detray/propagator/rk_stepper.hpp>
#include <detray/test/utils/simulation/random_scatterer.hpp>

// System include(s).
#include <limits>
#include <memory>

namespace traccc {

template <typename detector_t, typename bfield_t, typename track_generator_t,
          typename writer_t>
struct simulator {

    using scalar_type = typename detector_t::scalar_type;

    struct config {
        detray::propagation::config propagation;

        /// Particle hypothesis
        detray::pdg_particle<traccc::scalar> ptc_type =
            detray::muon<traccc::scalar>();
    };

    using algebra_type = typename detector_t::algebra_type;
    using bfield_type = bfield_t;

    using actor_chain_type =
        detray::actor_chain<detray::parameter_transporter<algebra_type>,
                            detray::random_scatterer<algebra_type>,
                            detray::parameter_resetter<algebra_type>, writer_t>;

    using navigator_type = detray::navigator<detector_t>;
    using stepper_type = detray::rk_stepper<
        typename bfield_type::view_t, algebra_type,
        detray::constrained_step<detray::dscalar<algebra_type>>>;
    using propagator_type =
        detray::propagator<stepper_type, navigator_type, actor_chain_type>;

    simulator(const detray::pdg_particle<scalar>& ptc_type, std::size_t events,
              const detector_t& det, const bfield_type& field,
              track_generator_t&& track_gen,
              typename writer_t::config&& writer_cfg,
              const std::string directory = "")
        : m_events(events),
          m_directory(directory),
          m_detector(det),
          m_field(field),
          m_track_generator(
              std::make_unique<track_generator_t>(std::move(track_gen))),
          m_writer_cfg(writer_cfg) {

        m_cfg.ptc_type = ptc_type;
        m_track_generator->config().charge(ptc_type.charge());
    }

    config& get_config() { return m_cfg; }

    void run() {

        for (std::size_t event_id = 0u; event_id < m_events; event_id++) {

            typename writer_t::state writer_state(
                event_id, std::move(m_writer_cfg), m_directory);

            // Set random seed
            m_scatterer.set_seed(event_id);
            writer_state.set_seed(event_id);

            auto actor_states = detray::tie(m_transporter, m_scatterer,
                                            m_resetter, writer_state);

            for (auto track : *m_track_generator.get()) {

                writer_state.write_particle(
                    track,
                    detail::correct_particle_hypothesis(m_cfg.ptc_type, track));

                typename propagator_type::state propagation(track, m_field,
                                                            m_detector);
                propagation.set_particle(
                    detail::correct_particle_hypothesis(m_cfg.ptc_type, track));

                propagator_type p(m_cfg.propagation);

                // Set overstep tolerance and stepper constraint
                propagation._stepping.template set_constraint<
                    detray::step::constraint::e_accuracy>(
                    m_cfg.propagation.stepping.step_constraint);

                p.propagate(propagation, actor_states);

                // Increase the particle id
                writer_state.particle_id++;
            }
        }
    }

    private:
    config m_cfg;
    std::size_t m_events{0u};
    std::string m_directory = "";
    const detector_t& m_detector;
    const bfield_type& m_field;
    std::unique_ptr<track_generator_t> m_track_generator;
    typename writer_t::config m_writer_cfg;

    /// Actor states
    typename detray::parameter_transporter<algebra_type>::state m_transporter{};
    typename detray::random_scatterer<algebra_type>::state m_scatterer{};
    typename detray::parameter_resetter<algebra_type>::state m_resetter{};
};

}  // namespace traccc
