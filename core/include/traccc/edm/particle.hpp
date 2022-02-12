/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "traccc/edm/track_parameters.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>

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

/// Container of particle for an event
template <template <typename> class vector_t>
using particle_collection = vector_t<particle>;

/// Convenience declaration for the particle collection type to use
/// in host code
using host_particle_collection = particle_collection<vecmem::vector>;

}  // namespace traccc