#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/internal_spacepoint.hpp"
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

inline void write_internal_spacepoints(
    size_t event,
    const traccc::host_internal_spacepoint_container &internal_sp_per_event) {

    traccc::internal_spacepoint_writer internal_spwriter{
        traccc::get_event_filename(event, "-internal_spacepoints.csv")};
    for (size_t i = 0; i < internal_sp_per_event.get_items().size(); ++i) {
        auto internal_sp_per_bin = internal_sp_per_event.get_items()[i];
        auto bin = internal_sp_per_event.get_headers()[i].global_index;

        for (const auto &internal_sp : internal_sp_per_bin) {
            const auto &x = internal_sp.m_x;
            const auto &y = internal_sp.m_y;
            const auto &z = internal_sp.m_z;
            const auto &varR = internal_sp.m_varianceR;
            const auto &varZ = internal_sp.m_varianceZ;
            internal_spwriter.append({bin, x, y, z, varR, varZ});
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
