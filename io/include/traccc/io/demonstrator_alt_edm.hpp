/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/alt_cell.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc {
struct alt_result {
    measurement_container_types::host measurements;
    spacepoint_container_types::host spacepoints;
};

using alt_demonstrator_input = vecmem::vector<alt_cell_reader_output_t>;
using alt_demonstrator_result = vecmem::vector<alt_result>;
}  // namespace traccc
