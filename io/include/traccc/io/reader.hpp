/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/digitization_config.hpp"
#include "traccc/io/binary.hpp"
#include "traccc/io/csv.hpp"
#include "traccc/io/data_format.hpp"
#include "traccc/io/demonstrator_edm.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <fstream>
#include <functional>

namespace traccc {

/// Function for spacepoint file reading. The output is traccc container.
///
/// @param event is the event index
/// @param measurements_directory is the directory of measurement file
/// @param data_format is the data format (e.g. csv or binary) of output file
/// @param resource is the vecmem resource
inline measurement_container_types::host read_measurements_from_event(
    size_t event, const std::string &measurements_directory,
    const traccc::data_format &data_format, vecmem::memory_resource &resource) {

    // Read the cells from the relevant event file
    if (data_format == traccc::data_format::csv) {
        std::string io_measurements_file =
            data_directory() + measurements_directory +
            get_event_filename(event, "-measurements.csv");
        traccc::measurement_reader mreader(
            io_measurements_file,
            {"geometry_id", "local_key", "local0", "local1", "phi", "theta",
             "time", "var_local0", "var_local1", "var_phi", "var_theta",
             "var_time"});
        return traccc::read_measurements(mreader, resource);
    } else if (data_format == traccc::data_format::binary) {
        std::string io_measurements_file =
            data_directory() + measurements_directory +
            get_event_filename(event, "-measurements.dat");

        vecmem::copy copy;

        return traccc::read_binary<measurement_container_types::host>(
            io_measurements_file, copy, resource);
    } else {
        throw std::invalid_argument("Allowed data format is csv or binary");
    }
}

inline traccc::demonstrator_input read(size_t events,
                                       const std::string &detector_file,
                                       const std::string &cell_directory,
                                       const std::string &digi_config_file,
                                       const traccc::data_format &data_format,
                                       vecmem::host_memory_resource &resource) {
    using namespace std::placeholders;
    auto geom = io::read_geometry(detector_file);
    auto digi_cfg = io::read_digitization_config(digi_config_file);
    traccc::demonstrator_input input_data(events, &resource);

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t event = 0; event < events; ++event) {
        traccc::cell_container_types::host cells_per_event = io::read_cells(
            event, cell_directory, data_format, &geom, &digi_cfg, &resource);

#if defined(_OPENMP)
#pragma omp critical
#endif
        { input_data[event] = cells_per_event; }
    }
    return input_data;
}

}  // namespace traccc
