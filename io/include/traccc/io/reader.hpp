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
