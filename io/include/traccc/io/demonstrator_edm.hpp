/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/cell.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc {
struct result {
    measurement_container_types::host measurements;
    spacepoint_container_types::host spacepoints;
};

using demonstrator_input = vecmem::vector<cell_container_types::host>;
using demonstrator_result = vecmem::vector<result>;
}  // namespace traccc
