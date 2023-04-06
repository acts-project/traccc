/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/alt_measurement.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::io {

/// Type definition for the reading of cells into a vector of cells and a
/// vector of modules. The cells hold a link to a position in the modules'
/// vector.
struct cell_reader_output {
    cell_collection_types::host cells;
    cell_module_collection_types::host modules;
};

/// Type definition for the reading of measurements into a vector of
/// alt_measurements and a vector of modules. The alt_measurements hold a link
/// to a position in the modules' vector.
struct measurement_reader_output {
    alt_measurement_collection_types::host measurements;
    cell_module_collection_types::host modules;
};

/// Type definition for the reading of spacepoints into a vector of spacepoitns
/// and a vector of modules. Each spacepoint holds a link to a position in the
/// modules' vector.
struct spacepoint_reader_output {
    spacepoint_collection_types::host spacepoints;
    cell_module_collection_types::host modules;
};

}  // namespace traccc::io
