/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// DFE include(s).
#include <dfe/dfe_namedtuple.hpp>

// System include(s).
#include <cstdint>

namespace traccc::io::csv {

/// Type to read information into about measurement<->hit association
struct measurement_hit_id {

    uint64_t measurement_id = 0;
    uint64_t hit_id = 0;

    DFE_NAMEDTUPLE(measurement_hit_id, measurement_id, hit_id);
};

}  // namespace traccc::io::csv
