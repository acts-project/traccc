/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/container.hpp"

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

/// Declare particle collection type
using particle_collection_types = collection_types<particle>;

}  // namespace traccc