#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"
#include "io/csv.hpp"
#include "io/demonstrator_edm.hpp"

namespace traccc {

inline void write_measurements(
    size_t event,
    const traccc::host_measurement_container &measurements_per_event) {
    traccc::measurement_writer mwriter{
        get_event_filename(event, "-measurements.csv")};
    for (size_t i = 0; i < measurements_per_event.items.size(); ++i) {
        auto measurements_per_module = measurements_per_event.items[i];
        auto module = measurements_per_event.headers[i];
        for (const auto &measurement : measurements_per_module) {
            const auto &local = measurement.local;
            mwriter.append({module.module, local[0], local[1], 0., 0.});
        }
    }
}

inline void write_spacepoints(
    size_t event,
    const traccc::host_spacepoint_container &spacepoints_per_event) {
    traccc::spacepoint_writer spwriter{
        get_event_filename(event, "-spacepoints.csv")};
    for (size_t i = 0; i < spacepoints_per_event.items.size(); ++i) {
        auto spacepoints_per_module = spacepoints_per_event.items[i];
        auto module = spacepoints_per_event.headers[i];
        for (const auto &spacepoint : spacepoints_per_module) {
            const auto &pos = spacepoint.global;
            spwriter.append({module, pos[0], pos[1], pos[2], 0., 0., 0.});
        }
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