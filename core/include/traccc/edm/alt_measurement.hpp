/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/cell.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/edm/measurement.hpp"

namespace traccc {

/// Alternative measurement structure which contains both the measurement and
/// a link to its module (held in a separate collection).
///
/// This can be used for storing all information in a single collection, whose
/// objects need to have both the header and item information from the
/// measurement container types.
struct alt_measurement {
    measurement meas;

    using link_type = cell_module_collection_types::view::size_type;
    link_type module_link;
};

/// Declare all alt measurement collection types
using alt_measurement_collection_types = collection_types<alt_measurement>;

}  // namespace traccc
