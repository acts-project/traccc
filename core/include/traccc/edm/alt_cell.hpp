/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// traccc include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"

namespace traccc {

/// Alternative cell structure which contains both the cell and a link to its
/// module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the cell
/// container types
struct alt_cell {
    cell c;
    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
};

/// Declare all cell collection types
using alt_cell_collection_types = collection_types<alt_cell>;

/// Type definition for the reading of cells into a vector of alt_cells and a
/// vector of modules. The alt_cells hold a link to a position in the modules'
/// vector.
struct alt_cell_reader_output_t {
    alt_cell_collection_types::host cells;
    cell_module_collection_types::host modules;
};

}  // namespace traccc
