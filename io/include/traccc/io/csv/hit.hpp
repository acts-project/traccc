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

/// Type to read information into about Fatras hits
struct hit {

    uint64_t particle_id = 0;
    uint64_t geometry_id = 0;
    float tx = 0;
    float ty = 0;
    float tz = 0;
    float tt = 0;
    float tpx = 0;
    float tpy = 0;
    float tpz = 0;
    float te = 0;
    float deltapx = 0;
    float deltapy = 0;
    float deltapz = 0;
    float deltae = 0;
    uint64_t index = 0;

    DFE_NAMEDTUPLE(hit, particle_id, geometry_id, tx, ty, tz, tt, tpx, tpy, tpz,
                   te, deltapx, deltapy, deltapz, deltae, index);
};

/// Type to read information into about Fatras hits with the new ACTS format
/// for particle IDs.
struct hit_with_split_particle_id {

    uint64_t particle_id_pv = 0;
    uint64_t particle_id_sv = 0;
    uint64_t particle_id_part = 0;
    uint64_t particle_id_gen = 0;
    uint64_t particle_id_subpart = 0;
    uint64_t geometry_id = 0;
    float tx = 0;
    float ty = 0;
    float tz = 0;
    float tt = 0;
    float tpx = 0;
    float tpy = 0;
    float tpz = 0;
    float te = 0;
    float deltapx = 0;
    float deltapy = 0;
    float deltapz = 0;
    float deltae = 0;
    uint64_t index = 0;

    DFE_NAMEDTUPLE(hit_with_split_particle_id, particle_id_pv, particle_id_sv,
                   particle_id_part, particle_id_gen, particle_id_subpart,
                   geometry_id, tx, ty, tz, tt, tpx, tpy, tpz, te, deltapx,
                   deltapy, deltapz, deltae, index);
};

}  // namespace traccc::io::csv
