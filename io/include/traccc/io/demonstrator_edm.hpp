/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/io/reader_edm.hpp"

namespace traccc {
struct result {
    alt_measurement_collection_types::host measurements;
    spacepoint_collection_types::host spacepoints;
};

using demonstrator_input = vecmem::vector<io::cell_reader_output>;
using demonstrator_result = vecmem::vector<result>;
}  // namespace traccc
