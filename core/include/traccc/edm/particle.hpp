/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <cstdint>

namespace traccc {

// Definition of truth particle
struct particle {
    uint64_t particle_id;
    int particle_type;
    int process;
    point3 pos;
    scalar time;
    vector3 mom;
    scalar mass;
    scalar charge;
};

inline bool operator<(const particle& lhs, const particle& rhs) {
    if (lhs.particle_id < rhs.particle_id) {
        return true;
    }
    return false;
}

/// Declare all particle collection types
using particle_collection_types = collection_types<particle>;

}  // namespace traccc
