/** Detray library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/io/csv/hit.hpp"
#include "traccc/io/csv/measurement.hpp"
#include "traccc/io/csv/measurement_hit_id.hpp"
#include "traccc/io/csv/particle.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/simulation/measurement_smearer.hpp"

// Detray core include(s).
#include "detray/definitions/qualifiers.hpp"
#include "detray/geometry/barcode.hpp"
#include "detray/geometry/surface.hpp"
#include "detray/propagator/base_actor.hpp"
#include "detray/tracks/bound_track_parameters.hpp"
#include "detray/tracks/free_track_parameters.hpp"

// Detray I/O include(s).
#include "detray/io/common/detail/utils.hpp"

// DFE include(s).
#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

namespace traccc {

template <typename smearer_t>
struct smearing_writer : detray::actor {

    using transform3_type = typename smearer_t::transform3_type;
    using scalar_type = typename transform3_type::scalar_type;

    using measurement_hit_id_writer =
        dfe::NamedTupleCsvWriter<io::csv::measurement_hit_id>;
    using measurement_writer = dfe::NamedTupleCsvWriter<io::csv::measurement>;
    using hit_writer = dfe::NamedTupleCsvWriter<io::csv::hit>;
    using particle_writer = dfe::NamedTupleCsvWriter<io::csv::particle>;

    struct config {
        smearer_t smearer;
    };

    struct state {
        state(std::size_t event_id, config&& writer_cfg,
              const std::string directory)
            : m_particle_writer(directory + traccc::io::get_event_filename(
                                                event_id, "-particles.csv")),
              m_hit_writer(directory + traccc::io::get_event_filename(
                                           event_id, "-hits.csv")),
              m_meas_writer(directory + traccc::io::get_event_filename(
                                            event_id, "-measurements.csv")),
              m_measurement_hit_id_writer(
                  directory + traccc::io::get_event_filename(
                                  event_id, "-measurement-simhit-map.csv")),
              m_meas_smearer(writer_cfg.smearer) {}

        uint64_t particle_id = 0u;
        particle_writer m_particle_writer;
        hit_writer m_hit_writer;
        measurement_writer m_meas_writer;
        measurement_hit_id_writer m_measurement_hit_id_writer;
        uint64_t m_hit_count = 0u;
        smearer_t m_meas_smearer;

        void set_seed(const uint_fast64_t sd) { m_meas_smearer.set_seed(sd); }

        void write_particle(
            const detray::free_track_parameters<transform3_type>& track) {
            io::csv::particle particle;
            const auto pos = track.pos();
            const auto mom = track.mom();

            particle.particle_id = particle_id;
            particle.vx = pos[0];
            particle.vy = pos[1];
            particle.vz = pos[2];
            particle.vt = track.time();
            particle.px = mom[0];
            particle.py = mom[1];
            particle.pz = mom[2];
            particle.q = track.charge();

            m_particle_writer.append(particle);
        }
    };

    struct measurement_kernel {

        template <typename mask_group_t, typename index_t>
        inline void operator()(
            const mask_group_t& mask_group, const index_t& index,
            const detray::bound_track_parameters<transform3_type>& bound_params,
            smearer_t& smearer, io::csv::measurement& iomeas) const {

            const auto& mask = mask_group[index];

            smearer(mask, smearer.get_offset(), bound_params, iomeas);
        }
    };

    template <typename propagator_state_t>
    void operator()(state& writer_state,
                    propagator_state_t& propagation) const {

        auto& navigation = propagation._navigation;
        auto& stepping = propagation._stepping;

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            // Write hits
            io::csv::hit hit;

            const auto track = stepping();
            const auto pos = track.pos();
            const auto mom = track.mom();

            const auto sf = navigation.get_surface();

            hit.particle_id = writer_state.particle_id;
            hit.geometry_id = sf.barcode().value();
            hit.tx = pos[0];
            hit.ty = pos[1];
            hit.tz = pos[2];
            hit.tt = track.time();
            hit.tpx = mom[0];
            hit.tpy = mom[1];
            hit.tpz = mom[2];

            writer_state.m_hit_writer.append(hit);

            // Write measurements
            io::csv::measurement meas;
            const auto bound_params = stepping._bound_params;

            meas.measurement_id = writer_state.m_hit_count;
            meas.geometry_id = hit.geometry_id;
            auto stddev_0 = writer_state.m_meas_smearer.stddev[0];
            auto stddev_1 = writer_state.m_meas_smearer.stddev[1];
            meas.var_local0 = stddev_0 * stddev_0;
            meas.var_local1 = stddev_1 * stddev_1;
            meas.phi = bound_params.phi();
            meas.theta = bound_params.theta();
            meas.time = bound_params.time();

            // Set local_key and smeared_local
            sf.template visit_mask<measurement_kernel>(
                bound_params, writer_state.m_meas_smearer, meas);

            writer_state.m_meas_writer.append(meas);

            // Write hit measurement map
            io::csv::measurement_hit_id measurement_hit_id;
            measurement_hit_id.hit_id = writer_state.m_hit_count;
            measurement_hit_id.measurement_id = writer_state.m_hit_count;
            writer_state.m_measurement_hit_id_writer.append(measurement_hit_id);
            writer_state.m_hit_count++;
        }
    }
};
}  // namespace traccc