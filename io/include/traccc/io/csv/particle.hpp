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
    float vx = 0.f;
    float vy = 0.f;
    float vz = 0.f;
    float vt = 0.f;
    float px = 0.f;
    float py = 0.f;
    float pz = 0.f;
    float m = 0.f;
    float q = 0.f;

    DFE_NAMEDTUPLE(particle, particle_id, particle_type, process, vx, vy, vz,
                   vt, px, py, pz, m, q);
};

/// Type used in reading CSV particle data into memory; new ACTS format with
/// split-up particle IDs
struct particle_with_split_id {

    uint64_t particle_id_pv = 0;
    uint64_t particle_id_sv = 0;
    uint64_t particle_id_part = 0;
    uint64_t particle_id_gen = 0;
    uint64_t particle_id_subpart = 0;
    int particle_type = 0;
    int process = 0;
    float vx = 0.f;
    float vy = 0.f;
    float vz = 0.f;
    float vt = 0.f;
    float px = 0.f;
    float py = 0.f;
    float pz = 0.f;
    float m = 0.f;
    float q = 0.f;

    DFE_NAMEDTUPLE(particle_with_split_id, particle_id_pv, particle_id_sv,
                   particle_id_part, particle_id_gen, particle_id_subpart,
                   particle_type, process, vx, vy, vz, vt, px, py, pz, m, q);
};

}  // namespace traccc::io::csv
