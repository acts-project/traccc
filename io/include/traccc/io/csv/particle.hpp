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

/// Type used in reading CSV particle data into memory
struct particle {

    uint64_t particle_id = 0;
    int particle_type = 0;
    int process = 0;
    float vx = 0.;
    float vy = 0.;
    float vz = 0.;
    float vt = 0.;
    float px = 0.;
    float py = 0.;
    float pz = 0.;
    float m = 0.;
    float q = 0.;

    DFE_NAMEDTUPLE(particle, particle_id, particle_type, process, vx, vy, vz,
                   vt, px, py, pz, m, q);
};

}  // namespace traccc::io::csv
