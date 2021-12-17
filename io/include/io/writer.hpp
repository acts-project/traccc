#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/seed.hpp"
#include "edm/spacepoint.hpp"
#include "io/csv.hpp"
#include "io/demonstrator_edm.hpp"
#include "io/utils.hpp"

namespace traccc {

inline void write_measurements(
    size_t event,
    const traccc::host_measurement_container &measurements_per_event) {
    traccc::measurement_writer mwriter{
        get_event_filename(event, "-measurements.csv")};
    for (size_t i = 0; i < measurements_per_event.size(); ++i) {
        auto measurements_per_module = measurements_per_event.get_items()[i];
        auto module = measurements_per_event.get_headers()[i];
        for (const auto &measurement : measurements_per_module) {
            const auto &local = measurement.local;
            mwriter.append({module.module, "", local[0], local[1], 0., 0., 0.,
                            0., 0., 0., 0., 0.});
        }
    }
}

inline void write_spacepoints(
    size_t event,
    const traccc::host_spacepoint_container &spacepoints_per_event) {
    traccc::spacepoint_writer spwriter{
        get_event_filename(event, "-spacepoints.csv")};
    for (size_t i = 0; i < spacepoints_per_event.size(); ++i) {
        auto spacepoints_per_module = spacepoints_per_event.get_items()[i];
        auto module = spacepoints_per_event.get_headers()[i];
        for (const auto &spacepoint : spacepoints_per_module) {
            const auto &pos = spacepoint.global;
            spwriter.append({module, pos[0], pos[1], pos[2], 0., 0., 0.});
        }
    }
}

inline void write_seeds(size_t event,
                        const traccc::host_seed_container &seeds) {
    traccc::seed_writer sd_writer{
        traccc::get_event_filename(event, "-seeds.csv")};
    for (auto &seed : seeds.get_items()[0]) {
        auto weight = seed.weight;
        auto z_vertex = seed.z_vertex;
        auto spB = seed.spB;
        auto spM = seed.spM;
        auto spT = seed.spT;

        sd_writer.append({weight, z_vertex, spB.x(), spB.y(), spB.z(), 0, 0,
                          spM.x(), spM.y(), spM.z(), 0, 0, spT.x(), spT.y(),
                          spT.z(), 0, 0});
    }
}

inline void write_estimated_track_parameters(
    size_t event,
    const traccc::host_bound_track_parameters_collection &params) {
    traccc::bound_track_parameters_writer btp_writer{
        traccc::get_event_filename(event, "-estimated_track_parameters.csv")};
    for (auto param : params) {
        auto &b_vec = param.vector();
        auto &b_cov = param.covariance();
        auto &surf_id = param.surface_id;

        btp_writer.append({b_vec[e_bound_loc0],   b_vec[e_bound_loc1],
                           b_vec[e_bound_theta],  b_vec[e_bound_phi],
                           b_vec[e_bound_qoverp], b_vec[e_bound_time],
                           b_cov(0, 0),           b_cov(0, 1),
                           b_cov(1, 1),           b_cov(0, 2),
                           b_cov(1, 2),           b_cov(2, 2),
                           b_cov(0, 3),           b_cov(1, 3),
                           b_cov(2, 3),           b_cov(3, 3),
                           b_cov(0, 4),           b_cov(1, 4),
                           b_cov(2, 4),           b_cov(3, 4),
                           b_cov(4, 4),           b_cov(0, 5),
                           b_cov(1, 5),           b_cov(2, 5),
                           b_cov(3, 5),           b_cov(4, 5),
                           b_cov(5, 5),           surf_id});
    }
}

inline void write(traccc::demonstrator_result aggregated_results) {

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t event = 0; event < aggregated_results.size(); ++event) {
        auto &eventResult = aggregated_results.at(event);
        write_measurements(event, eventResult.measurements);
        write_spacepoints(event, eventResult.spacepoints);
    }
}

}  // namespace traccc
