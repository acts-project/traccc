/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/simulation/smearing_writer.hpp"

// Detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"
#include "detray/simulation/random_scatterer.hpp"

// System include(s).
#include <limits>
#include <memory>

namespace traccc {

template <typename detector_t, typename bfield_t, typename track_generator_t,
          typename writer_t>
struct simulator {

    using scalar_type = typename detector_t::scalar_type;

    struct config {
        scalar_type overstep_tolerance{-10.f * detray::unit<scalar_type>::um};
        scalar_type step_constraint{std::numeric_limits<scalar_type>::max()};
    };

    using transform3 = typename detector_t::transform3;
    using bfield_type = bfield_t;

    using actor_chain_type =
        detray::actor_chain<dtuple, detray::parameter_transporter<transform3>,
                            detray::random_scatterer<transform3>,
                            detray::parameter_resetter<transform3>, writer_t>;

    using navigator_type = detray::navigator<detector_t>;
    using stepper_type =
        detray::rk_stepper<typename bfield_type::view_t, transform3,
                           detray::constrained_step<>>;
    using propagator_type =
        detray::propagator<stepper_type, navigator_type, actor_chain_type>;

    simulator(std::size_t events, const detector_t& det,
              const bfield_type& field, track_generator_t&& track_gen,
              typename writer_t::config&& writer_cfg,
              const std::string directory = "")
        : m_events(events),
          m_directory(directory),
          m_detector(det),
          m_field(field),
          m_track_generator(
              std::make_unique<track_generator_t>(std::move(track_gen))),
          m_writer_cfg(writer_cfg) {}

    config& get_config() { return m_cfg; }

    void run() {

        for (std::size_t event_id = 0u; event_id < m_events; event_id++) {

            typename writer_t::state writer_state(
                event_id, std::move(m_writer_cfg), m_directory);

            // Set random seed
            m_scatterer.set_seed(event_id);
            writer_state.set_seed(event_id);

            auto actor_states =
                std::tie(m_transporter, m_scatterer, m_resetter, writer_state);

            for (auto track : *m_track_generator.get()) {

                writer_state.write_particle(track);

                typename propagator_type::state propagation(track, m_field,
                                                            m_detector);

                propagator_type p({}, {});

                // Set overstep tolerance and stepper constraint
                propagation._stepping().set_overstep_tolerance(
                    m_cfg.overstep_tolerance);
                propagation._stepping.template set_constraint<
                    detray::step::constraint::e_accuracy>(
                    m_cfg.step_constraint);

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
    typename detray::parameter_transporter<transform3>::state m_transporter{};
    typename detray::random_scatterer<transform3>::state m_scatterer{};
    typename detray::parameter_resetter<transform3>::state m_resetter{};
};

}  // namespace traccc