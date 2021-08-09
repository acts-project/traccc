#pragma once

#include <map>

#include "edm/cell.hpp"
#include "edm/measurement.hpp"
#include "edm/spacepoint.hpp"

namespace traccc {
struct result {
    traccc::host_measurement_container measurements;
    traccc::host_spacepoint_container spacepoints;
};

using demonstrator_input = vecmem::vector<traccc::host_cell_container>;
using demonstrator_result = vecmem::vector<traccc::result>;
}  // namespace traccc