/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/qualifiers.hpp"
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

/// Comparison / ordering operator for (truth) particles
TRACCC_HOST_DEVICE
inline bool operator<(const particle& lhs, const particle& rhs) {

    return (lhs.particle_id < rhs.particle_id);
}

// Declare all (truth) particle collection types
TRACCC_DECLARE_COLLECTION_TYPES(particle);

}  // namespace traccc