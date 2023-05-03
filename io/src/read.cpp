/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read.hpp"

#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"

// OpenMP include(s).
#ifdef _OPENMP
#include <omp.h>
#endif

namespace traccc::io {

void read(demonstrator_input& out, std::size_t events,
          std::string_view directory, std::string_view detector_file,
          std::string_view digi_config_file, data_format format) {

    // Read in the detector configuration.
    const geometry geom = io::read_geometry(detector_file);
    const digitization_config digi_cfg =
        io::read_digitization_config(digi_config_file);

    assert(out.size() >= events);

    // Read in the cell data for all events. In parallel if possible.
#pragma omp parallel for
    for (std::size_t event = 0; event < events; ++event) {
        io::read_cells(out[event], event, directory, format, &geom, &digi_cfg);
    }
}

}  // namespace traccc::io
